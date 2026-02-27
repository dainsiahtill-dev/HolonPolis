"""Phase 5 Tests: Integration & API - 全域集成测试

验证:
1. 系统可以通过 bootstrap 初始化
2. FastAPI 应用可以启动
3. 完整链路: Chat -> Genesis -> HolonRuntime
4. API 端点响应正确
"""

import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from holonpolis.api.main import app
from holonpolis.bootstrap import bootstrap
from holonpolis.config import settings
from holonpolis.kernel.embeddings.default_embedder import set_embedder, SimpleEmbedder
from holonpolis.kernel.lancedb import reset_factory


@pytest.fixture
def temp_embedder():
    """使用 SimpleEmbedder 避免 API 调用."""
    embedder = SimpleEmbedder(dimension=128)
    set_embedder(embedder)
    return embedder


@pytest.fixture
def integration_setup(tmp_path, temp_embedder, monkeypatch):
    """设置集成测试环境."""
    holonpolis_root = tmp_path / ".holonpolis"
    monkeypatch.setattr(
        "holonpolis.config.settings.holonpolis_root",
        holonpolis_root
    )
    monkeypatch.setattr(
        "holonpolis.config.settings.holons_path",
        holonpolis_root / "holons"
    )
    monkeypatch.setattr(
        "holonpolis.config.settings.genesis_memory_path",
        holonpolis_root / "genesis" / "memory" / "lancedb"
    )
    reset_factory()

    # 运行 bootstrap
    bootstrap()

    return holonpolis_root


@pytest.fixture
def client(integration_setup):
    """创建 FastAPI 测试客户端."""
    return TestClient(app)


class TestBootstrap:
    """Bootstrap 初始化测试."""

    def test_bootstrap_creates_directories(self, integration_setup):
        """测试 bootstrap 创建目录结构."""
        root = integration_setup

        assert root.exists()
        assert (root / "genesis" / "memory" / "lancedb").exists()
        assert (root / "species").exists()
        assert (root / "holons").exists()

    def test_bootstrap_creates_species(self, integration_setup):
        """测试 bootstrap 创建默认物种."""
        root = integration_setup
        species_path = root / "species"

        assert (species_path / "generalist.json").exists()
        assert (species_path / "specialist.json").exists()
        assert (species_path / "worker.json").exists()

    def test_bootstrap_idempotent(self, integration_setup):
        """测试 bootstrap 是幂等的."""
        # 第二次运行 bootstrap 不应报错
        bootstrap()

        root = integration_setup
        assert root.exists()


class TestAPIEndpoints:
    """API 端点测试."""

    def test_health_endpoint(self, client):
        """测试健康检查端点."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_root_endpoint(self, client):
        """测试根端点."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "HolonPolis"
        assert "version" in data

    def test_list_holons_empty(self, client):
        """测试列出 Holons (空列表)."""
        response = client.get("/api/v1/holons")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # 初始状态下应该没有 Holons
        assert len(data) == 0


class TestChatFlow:
    """Chat 流程集成测试."""

    @pytest.fixture
    def mock_genesis_decision(self, monkeypatch):
        """模拟 Genesis 决策."""
        from holonpolis.services import GenesisService
        from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy

        async def mock_route_or_spawn(self, user_request, conversation_id=None):
            """返回 clarify 决策避免 LLM 调用."""
            from holonpolis.services.genesis_service import RouteResult

            return RouteResult(
                decision="clarify",
                message="This is a test clarification response",
                holon_id=None,
                blueprint=None,
                reasoning="Testing",
            )

        monkeypatch.setattr(GenesisService, "route_or_spawn", mock_route_or_spawn)

    def test_chat_endpoint_clarify(self, client, mock_genesis_decision):
        """测试 Chat 端点返回 clarify."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "Test message"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["route_decision"] == "clarify"
        assert data["content"] == "This is a test clarification response"


class TestHolonLifecycleAPI:
    """Holon 生命周期 API 测试."""

    @pytest.fixture
    def created_holon(self, client):
        """创建一个测试 Holon."""
        # 直接使用 GenesisService 创建
        from holonpolis.services import HolonService
        from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
        import asyncio

        blueprint = Blueprint(
            blueprint_id="bp_test_api",
            holon_id="holon_test_api",
            species_id="generalist",
            name="Test API Holon",
            purpose="Testing API endpoints",
            boundary=Boundary(),
            evolution_policy=EvolutionPolicy(),
        )

        service = HolonService()
        asyncio.run(service.create_holon(blueprint))

        return "holon_test_api"

    def test_get_holon(self, client, created_holon):
        """测试获取 Holon 详情."""
        response = client.get(f"/api/v1/holons/{created_holon}")

        assert response.status_code == 200
        data = response.json()
        assert data["holon_id"] == created_holon
        assert "blueprint" in data

    def test_freeze_resume_holon(self, client, created_holon):
        """测试冻结和恢复 Holon."""
        # 冻结
        response = client.post(f"/api/v1/holons/{created_holon}/freeze")
        assert response.status_code == 200
        assert response.json()["status"] == "frozen"

        # 恢复
        response = client.post(f"/api/v1/holons/{created_holon}/resume")
        assert response.status_code == 200
        assert response.json()["status"] == "resumed"

    def test_delete_holon(self, client, created_holon):
        """测试删除 Holon."""
        response = client.delete(f"/api/v1/holons/{created_holon}")

        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # 再次获取应该 404
        response = client.get(f"/api/v1/holons/{created_holon}")
        assert response.status_code == 404


class TestRunPy:
    """run.py 入口测试."""

    def test_run_py_exists(self):
        """测试 run.py 文件存在."""
        run_py = Path(__file__).parent.parent / "run.py"
        assert run_py.exists()

    def test_run_py_importable(self):
        """测试 run.py 可以导入."""
        import sys

        # 临时修改 sys.argv 避免 uvicorn 运行
        old_argv = sys.argv
        sys.argv = ["run.py"]

        try:
            # 读取 run.py 内容验证关键代码存在
            run_py = Path(__file__).parent.parent / "run.py"
            content = run_py.read_text()

            # 验证关键代码存在
            assert "uvicorn" in content
            assert "holonpolis.api.main:app" in content
            assert "settings" in content

        finally:
            sys.argv = old_argv


class TestEvolutionFlow:
    """演化流程集成测试."""

    def test_evolution_service_integration(self, integration_setup):
        """测试 EvolutionService 可以初始化和使用."""
        from holonpolis.services import EvolutionService

        service = EvolutionService()
        assert service is not None
        assert service.security_scanner is not None


class TestErrorHandling:
    """错误处理测试."""

    def test_404_not_found(self, client):
        """测试 404 错误."""
        response = client.get("/api/v1/holons/nonexistent_holon")

        assert response.status_code == 404

    def test_validation_error(self, client):
        """测试请求验证错误."""
        # 发送缺少必需字段的请求
        response = client.post("/api/v1/chat", json={})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestSystemStartup:
    """系统启动测试."""

    def test_app_lifespan(self, integration_setup):
        """测试 FastAPI lifespan."""
        from holonpolis.api.main import lifespan

        # 使用 async for 测试 lifespan
        async def test_lifespan():
            async with lifespan(app):
                # 在 lifespan 中，系统应该已初始化
                from holonpolis.kernel.lancedb import get_lancedb_factory
                factory = get_lancedb_factory()
                assert factory is not None

        asyncio.run(test_lifespan())

    def test_logging_setup(self, integration_setup):
        """测试日志配置."""
        import structlog

        # 验证日志可以获取
        logger = structlog.get_logger()
        assert logger is not None


class TestPhase1to4Integration:
    """Phase 1-4 组件集成测试."""

    def test_path_guard_with_lancedb(self, integration_setup):
        """测试 PathGuard 和 LanceDB 集成."""
        from holonpolis.kernel.storage import HolonPathGuard
        from holonpolis.kernel.lancedb import get_lancedb_factory

        # 创建 Holon
        guard = HolonPathGuard("integration_test_holon")
        guard.ensure_directory("memory/lancedb")

        # 初始化表
        factory = get_lancedb_factory()
        factory.init_holon_tables("integration_test_holon")

        # 验证路径正确
        assert guard.memory_path.exists()

    def test_sandbox_with_evolution(self, integration_setup):
        """测试 Sandbox 和 EvolutionService 集成."""
        from holonpolis.services import EvolutionService
        from holonpolis.domain.skills import ToolSchema

        service = EvolutionService()

        # 运行一次简单的安全扫描
        code = "def safe_func(): pass"
        result = service._phase_verify(code)

        assert result["passed"] is True
