import os
import click
import chromadb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from typing import Optional

console = Console()


def get_client():
    # try http client first, fallback to persistent
    if os.getenv('CHROMA_URL'):
        return chromadb.HttpClient(
            host=os.getenv('CHROMA_HOST', 'localhost'),
            port=int(os.getenv('CHROMA_PORT', '8000')),
            ssl=os.getenv('CHROMA_SSL', 'false').lower() == 'true'
        )
    return chromadb.PersistentClient(path="./chroma_data")


@click.group()
def cli():
    """ChromaDB Test Bench - Debug and manage collections"""
    pass


@cli.command()
@click.argument('name')
@click.option('--distance', type=click.Choice(['l2', 'ip', 'cosine']), default='l2', help='Distance metric')
def create(name: str, distance: str):
    """Create a new collection"""
    client = get_client()
    try:
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": distance}
        )
        console.print(f"[green]Created collection '{
                      name}' with {distance} distance metric")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")


@cli.command()
@click.argument('name')
def delete(name: str):
    """Delete a collection"""
    client = get_client()
    try:
        client.delete_collection(name)
        console.print(f"[green]Deleted collection '{name}'")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")


@cli.command(name='list')
def list_collections():
    """List all collections"""
    client = get_client()
    collections = client.list_collections()

    if not collections:
        console.print("[yellow]No collections found")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name")
    table.add_column("Distance")
    table.add_column("Count")

    for col in collections:
        collection = client.get_collection(col.name)
        table.add_row(
            col.name,
            col.metadata.get("hnsw:space", "l2"),
            str(collection.count())
        )

    console.print(table)


@cli.command()
@click.argument('name')
@click.option('--limit', default=10, help='Number of items to show')
def peek(name: str, limit: int):
    """Peek into a collection's contents"""
    client = get_client()
    try:
        collection = client.get_collection(name)
        results = collection.get(limit=limit)

        if not results['ids']:
            console.print("[yellow]Collection is empty")
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID")
        table.add_column("Metadata")
        table.add_column("Document Preview")

        for i in range(len(results['ids'])):
            doc_preview = results['documents'][i][:100] + "..." if len(
                results['documents'][i]) > 100 else results['documents'][i]
            metadata = JSON(str(results['metadatas'][i])
                            ) if results['metadatas'][i] else ""
            table.add_row(results['ids'][i], metadata, doc_preview)

        console.print(table)
        console.print(f"\nShowing {len(results['ids'])} of {
                      collection.count()} total items")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}")


@cli.command()
@click.argument('name')
@click.argument('query')
@click.option('--n-results', default=5, help='Number of results to return')
def search(name: str, query: str, n_results: int):
    """Search a collection with a text query"""
    client = get_client()
    try:
        collection = client.get_collection(name)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        if not results['ids']:
            console.print("[yellow]No results found")
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID")
        table.add_column("Distance")
        table.add_column("Document Preview")

        for i in range(len(results['ids'][0])):
            doc_preview = results['documents'][0][i][:100] + "..." if len(
                results['documents'][0][i]) > 100 else results['documents'][0][i]
            table.add_row(
                results['ids'][0][i],
                f"{results['distances'][0][i]:.4f}",
                doc_preview
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}")


@cli.command()
@click.argument('name')
def stats(name: str):
    """Show collection statistics"""
    client = get_client()
    try:
        collection = client.get_collection(name)
        results = collection.get(limit=1)

        if results['embeddings']:
            embedding_dim = len(results['embeddings'][0])
        else:
            embedding_dim = "Unknown (empty collection)"

        stats_table = Table(show_header=False, box=None)
        stats_table.add_row("Total Items", str(collection.count()))
        stats_table.add_row("Embedding Dimensions", str(embedding_dim))
        stats_table.add_row("Distance Metric",
                            collection.metadata.get("hnsw:space", "l2"))

        console.print(Panel(
            stats_table,
            title=f"Collection Stats: {name}",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"[red]Error: {str(e)}")


if __name__ == '__main__':
    cli()
