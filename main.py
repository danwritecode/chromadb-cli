import os
from pathlib import Path
import click
from chromadb import HttpClient, PersistentClient, Settings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from dotenv import load_dotenv

load_dotenv()

console = Console()


def load_config():
    """Load config from .env files with fallbacks"""
    # try loading from current directory
    if Path('.env').exists():
        load_dotenv('.env')
    # try loading from parent directory
    elif Path('../.env').exists():
        load_dotenv('../.env')
    # try loading from user home directory
    elif Path.home().joinpath('.eureka-chroma/.env').exists():
        load_dotenv(Path.home().joinpath('.eureka-chroma/.env'))


def get_client():
    """Get ChromaDB client"""
    host = os.getenv('CHROMA_HOST')
    token = os.getenv('CHROMA_TOKEN')
    if not host:
        console.print(
            "[yellow]No CHROMA_HOST set, using local persistent storage")
        return PersistentClient(path="./chroma_data")

    return HttpClient(
        host=host,
        port=int(os.getenv('CHROMA_PORT', '8000')),
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=token,
        )
    )


@click.group()
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """ChromaDB Test Bench - Debug and manage collections"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    if verbose:
        console.print(f"[blue]Environment:[/blue]")
        console.print(f"CHROMA_URL: {os.getenv('CHROMA_URL')}")
        console.print(f"CHROMA_HOST: {os.getenv('CHROMA_HOST')}")
        console.print(f"CHROMA_PORT: {os.getenv('CHROMA_PORT')}")
        console.print(f"CHROMA_SSL: {os.getenv('CHROMA_SSL')}")


@cli.command()
@click.argument('name')
@click.option('--distance', type=click.Choice(['l2', 'ip', 'cosine']), default='l2', help='Distance metric')
@click.option('--embedding-provider', type=click.Choice(['openai', 'cohere', 'huggingface', 'azure']),
              default='openai', help='Embedding model provider')
@click.option('--embedding-model', help='Specific embedding model to use (optional)')
def create(name: str, distance: str, embedding_provider: str, embedding_model: str):
    """Create a new collection with specified embedding provider"""
    client = get_client()

    # construct metadata with embedding info
    metadata = {
        "hnsw:space": distance,
        "embedding_provider": embedding_provider,
    }

    if embedding_model:
        metadata["embedding_model"] = embedding_model

    try:
        collection = client.create_collection(
            name=name,
            metadata=metadata
        )
        console.print(f"[green]Created collection '{name}':")
        console.print(f"  Distance metric: {distance}")
        console.print(f"  Embedding provider: {embedding_provider}")
        if embedding_model:
            console.print(f"  Embedding model: {embedding_model}")

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
    # table.add_column("Distance")
    table.add_column("Count")

    for col in collections:
        collection = client.get_collection(col.name)
        table.add_row(
            col.name,
            # col.metadata.get("hnsw:space", "l2"),
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
