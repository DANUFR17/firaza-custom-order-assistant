import os
import sys

# supaya `import api`, `import database`, dst. bisa dipakai dari folder tests/ (jalankan pytest dari ROOT proyek)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
