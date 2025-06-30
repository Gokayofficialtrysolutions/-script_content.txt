import sqlite3
import json
import datetime
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

class KnowledgeGraph:
    """
    Manages a SQLite-based knowledge graph to store and retrieve relationships
    between various entities like Knowledge Base items, topics, keywords, etc.
    It provides methods to initialize the schema, add nodes and edges,
    and query for related information.
    """
    def __init__(self, db_path: Path):
        """
        Initializes the KnowledgeGraph instance and connects to the SQLite database.

        Args:
            db_path (Path): The file path for the SQLite database.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False) # check_same_thread=False for potential async usage by orchestrator
        self.cursor = self.conn.cursor()
        self._initialize_schema()

    def _initialize_schema(self):
        """
        Initializes the database schema by creating tables if they don't already exist.
        """
        try:
            # kb_nodes table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS kb_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    content_preview TEXT,
                    created_timestamp_iso TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)

            # kb_edges table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS kb_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    weight REAL,
                    created_timestamp_iso TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (source_node_id) REFERENCES kb_nodes(node_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_node_id) REFERENCES kb_nodes(node_id) ON DELETE CASCADE
                )
            """)
            # Indexes for faster lookups on foreign keys and relationship types
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source_node ON kb_edges(source_node_id)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target_node ON kb_edges(target_node_id)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_relationship_type ON kb_edges(relationship_type)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_node_type ON kb_nodes(node_type)")

            self.conn.commit()
            print(f"KnowledgeGraph: Schema initialized/verified at {self.db_path}")
        except sqlite3.Error as e:
            print(f"KnowledgeGraph: Error initializing schema: {e}")
            # Potentially raise or handle more gracefully

    def add_node(self, node_id: str, node_type: str, content_preview: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Adds or updates (upserts) a node in the kb_nodes table.
        Returns True if successful, False otherwise.
        """
        timestamp = datetime.datetime.utcnow().isoformat()
        metadata_str = json.dumps(metadata) if metadata else None
        try:
            self.cursor.execute("""
                INSERT INTO kb_nodes (node_id, node_type, content_preview, created_timestamp_iso, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    node_type=excluded.node_type,
                    content_preview=excluded.content_preview,
                    metadata_json=excluded.metadata_json
            """, (node_id, node_type, content_preview, timestamp, metadata_str))
            self.conn.commit()
            # print(f"KnowledgeGraph: Node '{node_id}' (type: {node_type}) added/updated.")
            return True
        except sqlite3.Error as e:
            print(f"KnowledgeGraph: Error adding/updating node '{node_id}': {e}")
            return False

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a node by its ID."""
        try:
            self.cursor.execute("SELECT node_id, node_type, content_preview, created_timestamp_iso, metadata_json FROM kb_nodes WHERE node_id = ?", (node_id,))
            row = self.cursor.fetchone()
            if row:
                metadata = json.loads(row[4]) if row[4] else None
                return {"node_id": row[0], "node_type": row[1], "content_preview": row[2], "created_timestamp_iso": row[3], "metadata": metadata}
            return None
        except sqlite3.Error as e:
            print(f"KnowledgeGraph: Error getting node '{node_id}': {e}")
            return None

    def add_edge(self, source_node_id: str, target_node_id: str, relationship_type: str,
                 weight: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None, ensure_nodes: bool = False) -> Optional[str]:
        """
        Adds an edge to the kb_edges table.
        Generates a unique edge_id.
        If ensure_nodes is True, it will create minimal placeholder nodes if source or target don't exist.
        Returns the edge_id if successful, None otherwise.
        """
        if ensure_nodes:
            if not self.get_node(source_node_id):
                print(f"KnowledgeGraph: Source node '{source_node_id}' for edge not found. Creating placeholder.")
                self.add_node(source_node_id, node_type="UnknownPlaceholder", content_preview="Auto-created placeholder")
            if not self.get_node(target_node_id):
                print(f"KnowledgeGraph: Target node '{target_node_id}' for edge not found. Creating placeholder.")
                self.add_node(target_node_id, node_type="UnknownPlaceholder", content_preview="Auto-created placeholder")

        edge_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat()
        metadata_str = json.dumps(metadata) if metadata else None
        try:
            self.cursor.execute("""
                INSERT INTO kb_edges (edge_id, source_node_id, target_node_id, relationship_type, weight, created_timestamp_iso, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (edge_id, source_node_id, target_node_id, relationship_type, weight, timestamp, metadata_str))
            self.conn.commit()
            # print(f"KnowledgeGraph: Edge '{edge_id}' ({source_node_id} -> {target_node_id}, type: {relationship_type}) added.")
            return edge_id
        except sqlite3.IntegrityError as e: # Handles foreign key constraint violation if ensure_nodes=False and nodes don't exist
            print(f"KnowledgeGraph: Integrity error adding edge from '{source_node_id}' to '{target_node_id}': {e}. Ensure nodes exist or use ensure_nodes=True.")
            return None
        except sqlite3.Error as e:
            print(f"KnowledgeGraph: Error adding edge from '{source_node_id}' to '{target_node_id}': {e}")
            return None

    def get_edges(self, source_node_id: Optional[str] = None, target_node_id: Optional[str] = None, relationship_type: Optional[str] = None) -> list:
        """Retrieves edges based on criteria."""
        query = "SELECT edge_id, source_node_id, target_node_id, relationship_type, weight, created_timestamp_iso, metadata_json FROM kb_edges WHERE 1=1"
        params = []
        if source_node_id:
            query += " AND source_node_id = ?"
            params.append(source_node_id)
        if target_node_id:
            query += " AND target_node_id = ?"
            params.append(target_node_id)
        if relationship_type:
            query += " AND relationship_type = ?"
            params.append(relationship_type)

        results = []
        try:
            self.cursor.execute(query, params)
            for row in self.cursor.fetchall():
                metadata = json.loads(row[6]) if row[6] else None
                results.append({
                    "edge_id": row[0], "source_node_id": row[1], "target_node_id": row[2],
                    "relationship_type": row[3], "weight": row[4],
                    "created_timestamp_iso": row[5], "metadata": metadata
                })
        except sqlite3.Error as e:
            print(f"KnowledgeGraph: Error getting edges: {e}")
        return results

    def get_related_nodes(self, source_node_id: str, relationship_types: Optional[list[str]] = None, limit: int = 5) -> list:
        """
        Retrieves nodes related to a source_node_id via specified relationship types.
        Returns a list of target_node details.
        """
        related_nodes_data = []
        if not self.get_node(source_node_id): # Ensure source node exists
            # print(f"KnowledgeGraph: Source node '{source_node_id}' for get_related_nodes not found.")
            return []

        query = """
            SELECT DISTINCT n.node_id, n.node_type, n.content_preview, n.created_timestamp_iso, n.metadata_json, e.relationship_type
            FROM kb_nodes n
            JOIN kb_edges e ON n.node_id = e.target_node_id
            WHERE e.source_node_id = ?
        """
        params: list[Any] = [source_node_id]

        if relationship_types and len(relationship_types) > 0:
            placeholders = ','.join('?' for _ in relationship_types)
            query += f" AND e.relationship_type IN ({placeholders})"
            params.extend(relationship_types)

        query += " ORDER BY e.weight DESC, e.created_timestamp_iso DESC LIMIT ?" # Prioritize by weight then recency
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            for row in self.cursor.fetchall():
                metadata = json.loads(row[4]) if row[4] else None
                related_nodes_data.append({
                    "node_id": row[0], "node_type": row[1], "content_preview": row[2],
                    "created_timestamp_iso": row[3], "metadata": metadata,
                    "related_via": row[5] # The relationship type from the edge
                })
        except sqlite3.Error as e:
            print(f"KnowledgeGraph: Error in get_related_nodes for source '{source_node_id}': {e}")
        return related_nodes_data

    def get_source_nodes_related_to_target(self, target_node_id: str, relationship_types: Optional[list[str]] = None, limit: int = 5) -> list:
        """
        Retrieves source nodes related to a target_node_id via specified relationship types.
        Returns a list of source_node details.
        """
        source_nodes_data = []
        if not self.get_node(target_node_id): # Ensure target node exists
            # print(f"KnowledgeGraph: Target node '{target_node_id}' for get_source_nodes_related_to_target not found.")
            return []

        query = """
            SELECT DISTINCT n.node_id, n.node_type, n.content_preview, n.created_timestamp_iso, n.metadata_json, e.relationship_type
            FROM kb_nodes n
            JOIN kb_edges e ON n.node_id = e.source_node_id
            WHERE e.target_node_id = ?
        """
        params: list[Any] = [target_node_id]

        if relationship_types and len(relationship_types) > 0:
            placeholders = ','.join('?' for _ in relationship_types)
            query += f" AND e.relationship_type IN ({placeholders})"
            params.extend(relationship_types)

        query += " ORDER BY e.weight DESC, e.created_timestamp_iso DESC LIMIT ?" # Prioritize by weight then recency
        params.append(limit)

        try:
            self.cursor.execute(query, params)
            for row in self.cursor.fetchall():
                metadata = json.loads(row[4]) if row[4] else None
                source_nodes_data.append({
                    "node_id": row[0], "node_type": row[1], "content_preview": row[2],
                    "created_timestamp_iso": row[3], "metadata": metadata,
                    "related_to_target_via": row[5] # The relationship type from the edge
                })
        except sqlite3.Error as e:
            print(f"KnowledgeGraph: Error in get_source_nodes_related_to_target for target '{target_node_id}': {e}")
        return source_nodes_data

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print(f"KnowledgeGraph: Connection to {self.db_path} closed.")

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    db_file = Path("./test_terminus_graph.db")
    if db_file.exists():
        db_file.unlink() # Clean up before test

    kg = KnowledgeGraph(db_path=db_file)

    # Add nodes
    kg.add_node("doc1", "Document", "This is a test document about Python.", {"tags": ["python", "programming"], "author": "Jules"})
    kg.add_node("doc2", "Document", "Another document about web scraping.", {"source_url": "http://example.com"})
    kg.add_node("python_topic", "Topic", "Python Programming Language")
    kg.add_node("webscraping_topic", "Topic", "Web Scraping Techniques")

    # Test get_node
    retrieved_node = kg.get_node("doc1")
    print(f"Retrieved Node doc1: {retrieved_node}")

    # Add edges
    edge1_id = kg.add_edge("doc1", "python_topic", "HAS_TOPIC", weight=0.9)
    edge2_id = kg.add_edge("doc2", "webscraping_topic", "HAS_TOPIC", weight=0.8)
    edge3_id = kg.add_edge("doc1", "webscraping_topic", "MENTIONS_TOPIC", weight=0.3, metadata={"reason": "example usage"})
    edge4_id = kg.add_edge("non_existent_source", "python_topic", "RELATED_TO", ensure_nodes=True) # Test ensure_nodes

    print(f"Added edge 1 ID: {edge1_id}")
    print(f"Added edge 2 ID: {edge2_id}")
    print(f"Added edge 3 ID: {edge3_id}")
    print(f"Added edge 4 ID: {edge4_id}")

    # Test get_edges
    python_doc_edges = kg.get_edges(target_node_id="python_topic")
    print(f"Edges related to python_topic: {python_doc_edges}")

    doc1_source_edges = kg.get_edges(source_node_id="doc1")
    print(f"Edges originating from doc1: {doc1_source_edges}")

    # Test ON DELETE CASCADE (conceptual, requires manual DB check or more complex test)
    # kg.cursor.execute("DELETE FROM kb_nodes WHERE node_id = ?", ("python_topic",))
    # kg.conn.commit()
    # print(f"Edges after deleting python_topic: {kg.get_edges(target_node_id='python_topic')}") # Should be empty if cascade worked

    # Test get_related_nodes
    print("\nTesting get_related_nodes:")
    related_to_doc1 = kg.get_related_nodes("doc1", relationship_types=["HAS_TOPIC", "MENTIONS_TOPIC"], limit=2)
    print(f"Nodes related to doc1 (HAS_TOPIC, MENTIONS_TOPIC, limit 2): {json.dumps(related_to_doc1, indent=2)}")

    related_to_placeholder = kg.get_related_nodes("non_existent_source", limit=2)
    print(f"Nodes related to non_existent_source: {json.dumps(related_to_placeholder, indent=2)}")

    # Test get_source_nodes_related_to_target
    print("\nTesting get_source_nodes_related_to_target:")
    sources_for_python_topic = kg.get_source_nodes_related_to_target("python_topic", relationship_types=["HAS_TOPIC"], limit=2)
    print(f"Source nodes for python_topic (HAS_TOPIC, limit 2): {json.dumps(sources_for_python_topic, indent=2)}")
    # Expected: doc1 should be here if edge was doc1 -> python_topic (HAS_TOPIC)

    kg.close()
    if db_file.exists(): # Clean up after test
        db_file.unlink()
        print(f"Cleaned up {db_file}")

# To make this usable as a module:
# from .knowledge_graph import KnowledgeGraph (if in same package)
# or from src.core.knowledge_graph import KnowledgeGraph (if src is in PYTHONPATH)
# For now, direct import based on file structure might be:
# from knowledge_graph import KnowledgeGraph (if running from src/core)
# Or adjust path in orchestrator
