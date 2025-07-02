# TODO:

What's in flight atm?

# Add examples/tutorials
* Add a tutorial for resharding models

# Key functionality
* Make all operations atomic
* Support for non-blocking operations
* Support for mv/pop/peek/rm
* Support for key indexing
* RDMA Support
* Chunk send/recv

# Observability
* Add abilitiy to record back to the original requester

# Resharding support
* Test 'replicate' placement
* Test FSDP + TP (for TP support)
* Test FSDP + TP + EP
* Test HDSP + TP + EP

# SPMD support from monarch
* example using monarch without a single-controller
* Remove all async references, or make them optional

# Optimizations / Ideas
* Coalescieng on put/get
* Test non-tensor valeus
* Decrease number of copies
* Optimize 2x peak memory on full tensor assemble
* Re-consider "_has_full_tensor"
* Refactor DTensorPack to a PendingTensor class
* Reconsider assemble full tensor
* Add better handling for DTensor 'replicate' groups. (Avoid writting unnecessary replicates, exposing this as an option to the user)

# Random Things
* create a setup.py
* type hints + cleanup
* buck build fails with `error[E0425]: cannot find value `NCCL_SPLIT_NOCOLOR` in this scope`
* Search 'TODO's
* Parameterize tests
