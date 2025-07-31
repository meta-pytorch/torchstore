# TODO:

What's in flight atm?

# Add examples/tutorials
* Add a tutorial for resharding models

# Key functionality
* RDMA Support
* multi-node support
* Support for non-blocking operations (overlapping save/load) / trainer integrations
* Support for key indexing
* Support for mv/pop/peek/rm
* Make all operations atomic
* non-tensor valeus

# Observability
* Add abilitiy to record back to the original requester

# Resharding support
* Test FSDP + TP (for TP support)
* Test FSDP + TP + EP
* Test HDSP + TP + EP

# SPMD support from monarch
* Example using monarch without a single-controller
* Remove all async references, or make them optional

# Optimizations / Ideas
* Chunk send/recv
* Coalescieng multiple put/gets
* Decrease number of copies
* Optimize 2x peak memory on full tensor assemble
* Reconsider "_has_full_tensor"
* Reconsider assemble full tensor (allow users to fetch subsets before full tensor is written)
* Refactor DTensorPack to a PendingTensor class
* Add better handling for DTensor 'replicate' groups. (Avoid writting unnecessary replicates, exposing this as an option to the user)

# Random Things
* create a setup.py
* type hints + cleanup
* buck build fails with `error[E0425]: cannot find value `NCCL_SPLIT_NOCOLOR` in this scope`
* Search 'TODO's
* Parameterize tests
