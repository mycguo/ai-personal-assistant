#!/usr/bin/env python3
"""Test script for Milvus integration."""

import os
from dotenv import load_dotenv
from milvus_store import MilvusVectorStore

def test_milvus_integration():
    """Test basic Milvus operations."""
    load_dotenv()

    print("Testing Milvus integration...")
    print("-" * 50)

    # Initialize Milvus store
    print("1. Initializing Milvus store...")
    try:
        store = MilvusVectorStore(collection_name="test_collection")
        print("   ✓ Milvus store initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return False

    # Add test documents
    print("\n2. Adding test documents...")
    test_texts = [
        "This is a test document about artificial intelligence.",
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "Natural language processing helps computers understand human language.",
        "Vector databases are essential for similarity search."
    ]

    test_metadata = [
        {"source": "test", "topic": "AI"},
        {"source": "test", "topic": "ML"},
        {"source": "test", "topic": "DL"},
        {"source": "test", "topic": "NLP"},
        {"source": "test", "topic": "Vector DB"}
    ]

    try:
        ids = store.add_texts(test_texts, test_metadata)
        print(f"   ✓ Added {len(test_texts)} documents")
    except Exception as e:
        print(f"   ✗ Failed to add documents: {e}")
        return False

    # Test similarity search
    print("\n3. Testing similarity search...")
    test_query = "What is machine learning?"

    try:
        results = store.similarity_search(test_query, k=3)
        print(f"   ✓ Search completed. Found {len(results)} results")
        print("\n   Top results:")
        for i, doc in enumerate(results, 1):
            print(f"   {i}. {doc.page_content[:60]}...")
    except Exception as e:
        print(f"   ✗ Failed to search: {e}")
        return False

    # Get collection stats
    print("\n4. Getting collection statistics...")
    try:
        stats = store.get_collection_stats()
        if "error" not in stats:
            print(f"   ✓ Collection stats retrieved")
            print(f"   Number of entities: {stats.get('num_entities', 'N/A')}")
        else:
            print(f"   ⚠ Stats error: {stats['error']}")
    except Exception as e:
        print(f"   ✗ Failed to get stats: {e}")

    print("\n" + "-" * 50)
    print("✅ Milvus integration test completed successfully!")
    return True

if __name__ == "__main__":
    test_milvus_integration()