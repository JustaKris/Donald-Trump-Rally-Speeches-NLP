"""
Migration script to re-index documents with the new embedding model.

This script is needed because we upgraded from all-MiniLM-L6-v2 (384 dimensions)
to all-mpnet-base-v2 (768 dimensions). The existing ChromaDB collection needs
to be cleared and re-indexed with the new embeddings.

Run this script after updating the RAG service to use the new model.
"""

import sys
from pathlib import Path

from src.rag_service import RAGService

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def migrate_embeddings():
    """Clear and re-index documents with new embedding model."""
    print("=" * 70)
    print("RAG Service Migration: Re-indexing with improved embedding model")
    print("=" * 70)
    print()
    print("Changes:")
    print("  - Embedding model: all-MiniLM-L6-v2 → all-mpnet-base-v2")
    print("  - Dimensions: 384 → 768")
    print("  - Chunk size: 500 chars → 2048 chars (~512-768 tokens)")
    print("  - Chunk overlap: 50 chars → 150 chars")
    print()

    # Initialize RAG service with new settings
    print("Initializing RAG service with new configuration...")
    rag = RAGService()

    print("Configuration:")
    print(f"  - Embedding model: {rag.embedding_model}")
    print(f"  - Chunk size: {rag.chunk_size} chars")
    print(f"  - Chunk overlap: {rag.chunk_overlap} chars")
    print()

    # Clear existing collection
    print("Clearing existing collection...")
    success = rag.clear_collection()
    if success:
        print("✓ Collection cleared successfully")
    else:
        print("✗ Failed to clear collection")
        return False

    print()

    # Load documents with new embeddings
    print("Loading documents with new embeddings...")
    print("This may take a few minutes depending on the corpus size...")
    print()

    try:
        docs_loaded = rag.load_documents()
        print(f"✓ Successfully loaded {docs_loaded} documents")
    except Exception as e:
        print(f"✗ Error loading documents: {e}")
        return False

    print()

    # Get statistics
    print("Collection statistics:")
    stats = rag.get_stats()
    print(f"  - Total chunks: {stats['total_chunks']}")
    print(f"  - Unique sources: {stats['unique_sources']}")
    print(f"  - Embedding dimensions: {stats['embedding_model']}")
    print()

    print("=" * 70)
    print("Migration completed successfully!")
    print("=" * 70)
    print()
    print("The RAG service is now using:")
    print("  ✓ Better semantic understanding (mpnet-base-v2)")
    print("  ✓ Optimal chunk sizes (2048 chars / ~512-768 tokens)")
    print("  ✓ Improved retrieval with higher top_k defaults")
    print("  ✓ Enhanced confidence scoring")
    print("  ✓ Entity-aware retrieval and statistics")
    print()

    return True


if __name__ == "__main__":
    print()
    try:
        success = migrate_embeddings()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
