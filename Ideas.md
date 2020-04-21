Ideas
 - change test, train, valid split
 - dataset learnability by looking at spectral properties of bipartite graph (union of relations and entities weighted by # of training examples)
   - use synthetic data to check this?
 - global-local properties, i.e. how do graph properties change if we add the particular test example, etc.?

Experiments
 - graph properties per relation (train, train + valid, train + valid + test):
   + rank
   - number of subjects
   - number of objects
   - number of examples
   + largest singular value
   + symmetry ratio
   - number of connected components
   - average shortest path length (if available)
   - is_bipartite
 - union graph properties [(train, train + valid, train + valid + test); (raw, symmetrized)]:
   - number of connected components (check if certain connected components are harder to classify than others)
   - rank
   - symmetry ratio
   - eigenspectrum
   - singular value spectrum
   - fiedler vector partition
   - average shortest path length (if available)
