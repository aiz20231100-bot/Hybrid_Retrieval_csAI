
import arxiv

DATE_RANGE = (
    "submittedDate:[202101010000 TO 202610312359]"
)

TOPICS = {
    "Search and Planning": (
        'all:"automated planning" OR '
        'all:"heuristic search" OR '
        'all:"search-based planning"'
    ),

    "Knowledge Representation and Reasoning": (
        'all:"knowledge representation" OR '
        'all:"automated reasoning" OR '
        'all:"knowledge reasoning"'
    ),

    "Multi-Agent Systems": (
        'all:"multi-agent systems" OR '
        'all:"multi agent" OR '
        'all:"multiagent systems"'
    ),

    "Robotics and Embodied AI": (
        'all:"embodied AI" OR '
        'all:"robot learning" OR '
        'all:"robotic planning"'
    )
}


client = arxiv.Client(
    page_size=20,
    delay_seconds=3,
    num_retries=3
)


for topic, terms in TOPICS.items():

    query = f"cat:cs.AI AND ({terms}) AND {DATE_RANGE}"

    search = arxiv.Search(
        query=query,
        max_results=20,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = list(client.results(search))

    print("\n" + "=" * 70)
    print(f"TOPIC: {topic}")
    print(f"Sample results: {len(results)}")

    for i, paper in enumerate(results[:5], start=1):
        print(f"\n{i}. {paper.title}")
        print(f"ID: {paper.get_short_id()}")
        print(f"Link: {paper.entry_id}")
