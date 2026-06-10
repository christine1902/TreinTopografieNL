# TreinTopografieNL


Use in CLI:

install:
---
```pip install uv```

```uv pip install requests geopandas matplotlib shapely pillow numpy folium```

---
**Make map for students to study:**

```python .\script_to_make_learnmap.py --output_file "/output_file.jpg"```


---

**Make map to test the students:**

```python .\script_to_make_overhoormap.py --output_folder "path/to/output_folder" --versions 5 --ic 10 --spr 10```

--versions:

A maximum 26 versions (A-Z) will be used. Default number of versions is 3.

--ic:

The number of IC stations that will be put on the test. Default: 15

--spr:

The number of SPR stations that will be put on the test. Default: 5

---


backlog:
- clean this code up
- Better names for scripts
- Make sure the IC stations are really IC stations (or find a different source)
