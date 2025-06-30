import pytest
import sqlite3
from pathlib import Path
import json
import uuid
import time # For unique timestamps if needed rapidly

# Assuming knowledge_graph.py is in src/core/
# Adjust path if necessary based on how pytest is run and PYTHONPATH
from src.core.knowledge_graph import KnowledgeGraph

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path for each test."""
    db_file = tmp_path / f"test_kg_{uuid.uuid4()}.db"
    return db_file

@pytest.fixture
def kg_instance(temp_db_path: Path) -> KnowledgeGraph:
    """Fixture to create and tear down a KnowledgeGraph instance with a temporary DB."""
    kg = KnowledgeGraph(db_path=temp_db_path)
    yield kg # Provide the instance to the test
    kg.close()
    # No need to manually delete temp_db_path, tmp_path fixture handles it

@pytest.fixture
def kg_in_memory() -> KnowledgeGraph:
    """Fixture for an in-memory KnowledgeGraph instance."""
    # Using ":memory:" creates a new DB for each connection,
    # so ensure it's the same instance if multiple methods in a test use it.
    # For most test cases, a fresh in-memory DB per test function is fine.
    kg = KnowledgeGraph(db_path=Path(":memory:"))
    yield kg
    kg.close()


class TestKnowledgeGraphSchema:
    def test_initialization_creates_tables(self, kg_in_memory: KnowledgeGraph):
        kg = kg_in_memory # Use the in-memory version for schema checks
        cursor = kg.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kb_nodes'")
        assert cursor.fetchone() is not None, "kb_nodes table should exist"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kb_edges'")
        assert cursor.fetchone() is not None, "kb_edges table should exist"

    def test_table_columns_nodes(self, kg_in_memory: KnowledgeGraph):
        kg = kg_in_memory
        cursor = kg.conn.cursor()
        cursor.execute("PRAGMA table_info(kb_nodes)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        assert "node_id" in columns and columns["node_id"] == "TEXT"
        assert "node_type" in columns and columns["node_type"] == "TEXT"
        assert "content_preview" in columns and columns["content_preview"] == "TEXT"
        assert "created_timestamp_iso" in columns and columns["created_timestamp_iso"] == "TEXT"
        assert "metadata_json" in columns and columns["metadata_json"] == "TEXT"

        # Check primary key
        cursor.execute("SELECT COUNT(*) FROM pragma_table_info('kb_nodes') WHERE pk > 0 AND name = 'node_id'")
        assert cursor.fetchone()[0] == 1, "node_id should be the primary key"


    def test_table_columns_edges(self, kg_in_memory: KnowledgeGraph):
        kg = kg_in_memory
        cursor = kg.conn.cursor()
        cursor.execute("PRAGMA table_info(kb_edges)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        assert "edge_id" in columns and columns["edge_id"] == "TEXT"
        assert "source_node_id" in columns and columns["source_node_id"] == "TEXT"
        assert "target_node_id" in columns and columns["target_node_id"] == "TEXT"
        assert "relationship_type" in columns and columns["relationship_type"] == "TEXT"
        assert "weight" in columns and columns["weight"] == "REAL"
        assert "created_timestamp_iso" in columns and columns["created_timestamp_iso"] == "TEXT"
        assert "metadata_json" in columns and columns["metadata_json"] == "TEXT"

        # Check primary key
        cursor.execute("SELECT COUNT(*) FROM pragma_table_info('kb_edges') WHERE pk > 0 AND name = 'edge_id'")
        assert cursor.fetchone()[0] == 1, "edge_id should be the primary key"

        # Check Foreign Keys (more involved, might need to check pragma_foreign_key_list)
        # For simplicity, we'll trust the CREATE TABLE statement for now. A deeper test could parse pragma_foreign_key_list.


class TestNodeOperations:
    def test_add_and_get_node_simple(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        node_id = "node1"
        node_type = "TestDocument"
        assert kg.add_node(node_id, node_type, "Preview content") == True

        retrieved = kg.get_node(node_id)
        assert retrieved is not None
        assert retrieved["node_id"] == node_id
        assert retrieved["node_type"] == node_type
        assert retrieved["content_preview"] == "Preview content"
        assert retrieved["metadata"] is None
        assert "created_timestamp_iso" in retrieved

    def test_add_and_get_node_with_metadata(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        node_id = "node_meta"
        node_type = "MetadataNode"
        metadata = {"key1": "value1", "num_key": 123}
        assert kg.add_node(node_id, node_type, metadata=metadata) == True

        retrieved = kg.get_node(node_id)
        assert retrieved is not None
        assert retrieved["metadata"] == metadata

    def test_update_node_on_conflict(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        node_id = "node_update"
        kg.add_node(node_id, "OriginalType", "Original Preview")

        # Update with new type and preview
        assert kg.add_node(node_id, "UpdatedType", "Updated Preview", metadata={"new_field": True}) == True

        retrieved = kg.get_node(node_id)
        assert retrieved["node_type"] == "UpdatedType"
        assert retrieved["content_preview"] == "Updated Preview"
        assert retrieved["metadata"] == {"new_field": True}

    def test_get_non_existent_node(self, kg_instance: KnowledgeGraph):
        assert kg_instance.get_node("non_existent_node_id") is None

class TestEdgeOperations:
    def test_add_and_get_edge_simple(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        src_id, tgt_id = "src_node_simple", "tgt_node_simple"
        kg.add_node(src_id, "Source")
        kg.add_node(tgt_id, "Target")

        edge_id = kg.add_edge(src_id, tgt_id, "RELATES_TO")
        assert edge_id is not None

        edges = kg.get_edges(source_node_id=src_id, target_node_id=tgt_id, relationship_type="RELATES_TO")
        assert len(edges) == 1
        edge = edges[0]
        assert edge["edge_id"] == edge_id
        assert edge["source_node_id"] == src_id
        assert edge["target_node_id"] == tgt_id
        assert edge["relationship_type"] == "RELATES_TO"
        assert edge["weight"] is None
        assert edge["metadata"] is None

    def test_add_edge_with_weight_and_metadata(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        src_id, tgt_id = "src_node_meta_edge", "tgt_node_meta_edge"
        kg.add_node(src_id, "Source")
        kg.add_node(tgt_id, "Target")
        metadata = {"reason": "testing"}

        edge_id = kg.add_edge(src_id, tgt_id, "HAS_PROPERTY", weight=0.75, metadata=metadata)
        assert edge_id is not None

        edges = kg.get_edges(edge_id=edge_id) # Assuming get_edges can filter by edge_id (it cannot directly based on current sig)
                                            # Let's get all and filter
        all_edges = kg.get_edges(source_node_id=src_id)
        edge = next((e for e in all_edges if e["edge_id"] == edge_id), None)

        assert edge is not None
        assert edge["weight"] == 0.75
        assert edge["metadata"] == metadata

    def test_add_edge_ensure_nodes_creates_placeholders(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        src_id_new, tgt_id_new = "new_src_placeholder", "new_tgt_placeholder"

        edge_id = kg.add_edge(src_id_new, tgt_id_new, "LINKS_NEW", ensure_nodes=True)
        assert edge_id is not None

        assert kg.get_node(src_id_new) is not None
        assert kg.get_node(src_id_new)["node_type"] == "UnknownPlaceholder"
        assert kg.get_node(tgt_id_new) is not None
        assert kg.get_node(tgt_id_new)["node_type"] == "UnknownPlaceholder"

    def test_add_edge_fails_if_nodes_do_not_exist_and_ensure_nodes_false(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        # Do not create nodes "src_fail" and "tgt_fail"
        edge_id = kg.add_edge("src_fail", "tgt_fail", "FAILS_LINK", ensure_nodes=False)
        assert edge_id is None # Should fail due to foreign key constraint

    def test_get_edges_filtering(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        s1, t1, t2 = "s1_filter", "t1_filter", "t2_filter"
        kg.add_node(s1, "Doc"); kg.add_node(t1, "Topic"); kg.add_node(t2, "Keyword")

        kg.add_edge(s1, t1, "HAS_TOPIC")
        kg.add_edge(s1, t2, "HAS_KEYWORD")
        kg.add_edge(t1, s1, "IS_TOPIC_OF") # Different direction

        assert len(kg.get_edges(source_node_id=s1)) == 2
        assert len(kg.get_edges(target_node_id=t1)) == 1
        assert len(kg.get_edges(relationship_type="HAS_TOPIC")) == 1
        assert len(kg.get_edges(source_node_id=s1, relationship_type="HAS_KEYWORD")) == 1
        assert len(kg.get_edges(source_node_id="non_existent")) == 0


class TestRelatedNodesQueries:
    @pytest.fixture(autouse=True)
    def setup_graph_data(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        # Nodes
        kg.add_node("docA", "Document", "Content about AI and Python")
        kg.add_node("docB", "Document", "Content about Python and Web")
        kg.add_node("docC", "Document", "Content about AI Ethics")
        kg.add_node("topic_ai", "Topic", "Artificial Intelligence")
        kg.add_node("topic_python", "Topic", "Python Programming")
        kg.add_node("topic_web", "Topic", "Web Development")
        kg.add_node("keyword_ethics", "Keyword", "ethics")

        # Edges (doc -> topic/keyword)
        kg.add_edge("docA", "topic_ai", "HAS_TOPIC", weight=0.9)
        kg.add_edge("docA", "topic_python", "HAS_TOPIC", weight=0.8)
        kg.add_edge("docB", "topic_python", "HAS_TOPIC", weight=0.9)
        kg.add_edge("docB", "topic_web", "HAS_TOPIC", weight=0.7)
        kg.add_edge("docC", "topic_ai", "HAS_TOPIC", weight=0.6)
        kg.add_edge("docC", "keyword_ethics", "HAS_KEYWORD", weight=0.95)
        # For get_related_nodes (topic -> doc, if such relationships were added)
        # kg.add_edge("topic_ai", "docA", "IS_TOPIC_OF_DOC")

    def test_get_related_nodes(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        # To test get_related_nodes, we need edges where the source is a topic/keyword
        # and target is a document. The current setup_graph_data has doc -> topic.
        # Let's add a reverse relationship for testing this specific function.
        kg.add_edge("topic_python", "docA", "TOPIC_APPEARS_IN_DOC", weight=0.8)
        kg.add_edge("topic_python", "docB", "TOPIC_APPEARS_IN_DOC", weight=0.9)

        related_to_python_topic = kg.get_related_nodes("topic_python", relationship_types=["TOPIC_APPEARS_IN_DOC"])
        assert len(related_to_python_topic) == 2
        node_ids_related = {n["node_id"] for n in related_to_python_topic}
        assert "docA" in node_ids_related
        assert "docB" in node_ids_related

        # Check ordering (docB should be first due to higher weight)
        assert related_to_python_topic[0]["node_id"] == "docB"

        related_to_ai_limit1 = kg.get_related_nodes("topic_ai", limit=1) # Assuming we add TOPIC_APPEARS_IN_DOC for AI too
        kg.add_edge("topic_ai", "docA", "TOPIC_APPEARS_IN_DOC", weight=0.9)
        kg.add_edge("topic_ai", "docC", "TOPIC_APPEARS_IN_DOC", weight=0.6)
        related_to_ai_limit1_actual = kg.get_related_nodes("topic_ai", relationship_types=["TOPIC_APPEARS_IN_DOC"], limit=1)
        assert len(related_to_ai_limit1_actual) == 1
        assert related_to_ai_limit1_actual[0]["node_id"] == "docA"


    def test_get_source_nodes_related_to_target(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        sources_for_topic_ai = kg.get_source_nodes_related_to_target("topic_ai", relationship_types=["HAS_TOPIC"])
        assert len(sources_for_topic_ai) == 2 # docA, docC
        node_ids_sources_ai = {n["node_id"] for n in sources_for_topic_ai}
        assert "docA" in node_ids_sources_ai
        assert "docC" in node_ids_sources_ai
        # docA (0.9) should come before docC (0.6)
        assert sources_for_topic_ai[0]["node_id"] == "docA"

        sources_for_keyword_ethics = kg.get_source_nodes_related_to_target("keyword_ethics", relationship_types=["HAS_KEYWORD"], limit=1)
        assert len(sources_for_keyword_ethics) == 1
        assert sources_for_keyword_ethics[0]["node_id"] == "docC"

    def test_get_related_nodes_no_relations(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        # topic_web is a target in setup, but not a source for TOPIC_APPEARS_IN_DOC
        assert len(kg.get_related_nodes("topic_web", relationship_types=["TOPIC_APPEARS_IN_DOC"])) == 0

    def test_get_source_nodes_no_relations(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        # docA is a source, not a target for HAS_TOPIC
        assert len(kg.get_source_nodes_related_to_target("docA", relationship_types=["HAS_TOPIC"])) == 0


class TestCascadeDelete:
    def test_on_delete_cascade_source_node(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        s_cas, t_cas = "s_cascade", "t_cascade"
        kg.add_node(s_cas, "DocSource")
        kg.add_node(t_cas, "DocTarget")
        edge_id = kg.add_edge(s_cas, t_cas, "LINKS_TO")
        assert edge_id is not None
        assert len(kg.get_edges(source_node_id=s_cas)) == 1

        # Delete source node
        kg.cursor.execute("DELETE FROM kb_nodes WHERE node_id = ?", (s_cas,))
        kg.conn.commit()

        assert kg.get_node(s_cas) is None
        # Edge should be gone due to ON DELETE CASCADE for source_node_id
        assert len(kg.get_edges(source_node_id=s_cas)) == 0
        assert len(kg.get_edges(edge_id=edge_id)) == 0 # Check specifically if the edge is gone
        assert kg.get_node(t_cas) is not None # Target node should still exist

    def test_on_delete_cascade_target_node(self, kg_instance: KnowledgeGraph):
        kg = kg_instance
        s_cas2, t_cas2 = "s_cascade2", "t_cascade2"
        kg.add_node(s_cas2, "DocSource")
        kg.add_node(t_cas2, "DocTarget")
        edge_id = kg.add_edge(s_cas2, t_cas2, "POINTS_AT")
        assert edge_id is not None
        assert len(kg.get_edges(target_node_id=t_cas2)) == 1

        # Delete target node
        kg.cursor.execute("DELETE FROM kb_nodes WHERE node_id = ?", (t_cas2,))
        kg.conn.commit()

        assert kg.get_node(t_cas2) is None
        # Edge should be gone due to ON DELETE CASCADE for target_node_id
        assert len(kg.get_edges(target_node_id=t_cas2)) == 0
        assert len(kg.get_edges(edge_id=edge_id)) == 0
        assert kg.get_node(s_cas2) is not None # Source node should still exist

# Example of how to run with pytest:
# Ensure pytest and pytest-asyncio are installed
# From the root of the project (where src/ and tests/ are):
# PYTHONPATH=./src pytest tests/core/test_knowledge_graph.py
# Or if src is already in PYTHONPATH:
# pytest tests/core/test_knowledge_graph.py
#
# If using VSCode, configure testing to use pytest and it should discover tests.
# Ensure your launch.json or settings.json for VSCode correctly sets PYTHONPATH if needed.
# Example .vscode/settings.json:
# {
# "python.testing.pytestArgs": [
# "tests"
# ],
# "python.testing.unittestEnabled": false,
# "python.testing.pytestEnabled": true,
# "python.envFile": "${workspaceFolder}/.env" // if you use .env for PYTHONPATH
# }
# Example .env file:
# PYTHONPATH="${PYTHONPATH}:${workspaceFolder}/src"
#
# For this specific structure, if you run pytest from the repository root,
# you might need to tell Python where 'src' is.
# One way is to add `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`
# at the top of this test file before `from src.core...`, but using PYTHONPATH is cleaner.
# For now, assuming pytest is run with src in path.

# A simple conftest.py in tests/ directory could also help with path manipulation if needed:
# tests/conftest.py
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# This makes `src` importable for all tests.
