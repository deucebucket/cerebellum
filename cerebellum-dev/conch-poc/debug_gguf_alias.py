import gguf
import numpy as np
import os

# Create a small dummy GGUF
fname = "test_alias.gguf"
writer = gguf.GGUFWriter(fname, "dummy")

data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
writer.add_tensor("tensor.original", data)
# We can't easily add an alias with the high-level writer
# It usually appends data.
# We'll have to use the low-level reader/writer or manual surgery.

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

print(f"Created {fname}")

# Now try to read and see the offset
reader = gguf.GGUFReader(fname)
t = reader.get_tensor_by_name("tensor.original")
print(f"Original: offset={t.offset}, size={t.n_bytes}")

# Surgery: Add a new tensor metadata pointing to the same offset
# This requires rewriting the header/metadata.
