from localization import VisualLocalizer

ref_dir = r"D:\python\mapanything_cs\photo\effiel"
query_image = r"D:\python\mapanything_cs\query\1.jpg"

loc = VisualLocalizer(ref_dir)

print("status：")
print(loc.get_status())

with open(query_image, "rb") as f:
    result = loc.localize_image_bytes(f.read())

print("localization：")
print(result)