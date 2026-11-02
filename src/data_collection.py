
import arxiv


def test_arxiv_connection():

    print("Connecting to arXiv...")

    client = arxiv.Client(
        page_size=5,
        delay_seconds=3,
        num_retries=3
    )

    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=5,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    results = client.results(search)

    for index, paper in enumerate(results, start=1):

        print("\n" + "=" * 60)

        print(f"Paper number: {index}")
        print(f"Title: {paper.title}")
        print(f"ArXiv ID: {paper.get_short_id()}")
        print(f"Published: {paper.published.date()}")
        print(f"Categories: {paper.categories}")
        print(f"Link: {paper.entry_id}")
        print(f"Abstract: {paper.summary[:250]}")

    print("\nConnection test completed!")


if __name__ == "__main__":
    test_arxiv_connection()
