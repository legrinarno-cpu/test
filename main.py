        dirs[:] = sorted(name for name in dirs if not name.startswith(".")
                         and name != "__MACOSX" and not (parent / name).is_symlink())
        for name in names:
            path = parent / name
            if (not name.startswith(".") and path.suffix.lower() == ".txt"
                    and path.is_file() and not path.is_symlink()):
                files.append(path)

    documents = []
    # Sorting filenames gives each document a stable row number in the matrix.
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")  # Older Western-language text files.
        documents.append(Document(path.relative_to(root).as_posix(), text))
    if not documents:
        raise ValueError("The corpus folder contains no .txt documents.")
    return documents


def create_tf_matrix(documents: list[Document]):
    """Build raw term counts and an inverted index: term -> {row: count}."""
    inverted_index = defaultdict(dict)
    for row, document in enumerate(documents):
        # Counter counts each processed word within this one document.
        for term, count in Counter(preprocess(document.text)).items():
            inverted_index[term][row] = count

    # Example: {"war": {0: 3, 4: 1}} means "war" occurs in documents 0 and 4.
    # Same rows/columns as the lecture's pandas table; implicit zeros save RAM.
    columns = {
        term: pd.arrays.SparseArray(
            [inverted_index[term].get(row, 0) for row in range(len(documents))],
            dtype=pd.SparseDtype("int64", 0),
        )
        for term in sorted(inverted_index)
    }
    tf_matrix = pd.DataFrame(columns, index=range(len(documents)))
    return tf_matrix, dict(inverted_index)


def tf_to_tfidf(tf_matrix):
    """The lecturer's exact formula; no extra smoothing or normalization."""
    if not len(tf_matrix):
        raise ValueError("At least one document is required.")
    columns = {}
    for term in tf_matrix.columns:
        # N is the document count; df counts documents containing this term.
        # IDF gives rare terms more weight than terms found in many documents.
        document_frequency = tf_matrix[term].sparse.npoints
        idf = log(len(tf_matrix) / (1 + document_frequency))
        # Scalar multiplication keeps each column's zeros implicit.
        columns[term] = tf_matrix[term].array * idf
    return pd.DataFrame(columns, index=tf_matrix.index)


def queryterm(term: str, tfidf_matrix, inverted_index) -> dict[int, float]:
    """Retrieve only indexed matches, including zero or negative TF-IDF."""
    if term not in inverted_index:
        return {}
    column = tfidf_matrix[term]
    return {row: float(column.iloc[row]) for row in inverted_index[term]}


def querystring(query: str, tfidf_matrix, documents, inverted_index):
    """OR-match distinct query terms; sum their TF-IDF weights to rank."""
    if not query.strip():
        raise ValueError("The query must contain at least one non-whitespace character.")
    contributions = defaultdict(list)
    # set() avoids counting a repeated query word twice after stemming.
    for term in sorted(set(preprocess(query))):
        for row, score in queryterm(term, tfidf_matrix, inverted_index).items():
            contributions[row].append(score)

    # Add each matching query term's weight to obtain one score per document.
    results = [SearchResult(documents[row].path, fsum(scores), documents[row].text[:200])
               for row, scores in contributions.items()]
    # Highest score first; alphabetical filenames make equal scores predictable.
    return sorted(results, key=lambda result: (-result.score, result.path))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("query", help='Multi-word query, e.g. "star wars Lucas"')
    parser.add_argument("corpus", help="Folder containing the .txt documents")
    args = parser.parse_args(argv)
    try:
        if not args.query.strip():
            raise ValueError("The query must contain at least one non-whitespace character.")
        # Pipeline: read the corpus -> count words -> weight them -> run the query.
        documents = load_documents(args.corpus)
        tf_matrix, inverted_index = create_tf_matrix(documents)
        tfidf_matrix = tf_to_tfidf(tf_matrix)
        results = querystring(args.query, tfidf_matrix, documents, inverted_index)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if not results:
        print("No matching documents. Try words other than punctuation or English stopwords.")
        return 0
    print(f"{len(results)} matching documents from {len(documents)} indexed documents.")
    for rank, result in enumerate(results, 1):
        print(f"\n{rank}. {result.path} | TF-IDF score: {result.score:.12g}")
        print(result.preview)  # Exactly the first 200 original characters (or fewer).
    return 0


if __name__ == "__main__":
    # Run the CLI when this file is launched directly with Python.
    raise SystemExit(main())
