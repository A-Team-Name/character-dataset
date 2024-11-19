from yaml import load, YAMLObject, Loader
import random

class Config:
    CHARACTER_CODES = list(map(lambda x: x.strip(), """u2190
        u2191
        u2192
        u2193
        u2206
        u220a
        u2218
        u221a
        u2227
        u2228
        u2229
        u222a
        u2260
        u2261
        u2262
        u2264
        u2265
        u2282
        u2283
        u2286
        u2287
        u2296
        u22a2
        u22a3
        u22a4
        u22a5
        u2308
        u230a
        u2336
        u2337
        u2338
        u2339
        u233a
        u233d
        u233f
        u2340
        u2349
        u234b
        u234e
        u2352
        u2355
        u2359
        u235b
        u235f
        u2360
        u2362
        u2363
        u2364
        u2365
        u2368
        u236a
        u236b
        u236c
        u2371
        u2372
        u2373
        u2374
        u2377
        u2378
        u25cb
        u30
        u31
        u32
        u33
        u34
        u35
        u36
        u37
        u38
        u39
        u41
        u42
        u43
        u44
        u45
        u46
        u47
        u48
        u49
        u4a
        u4b
        u4c
        u4d
        u4e
        u4f
        u50
        u51
        u52
        u53
        u54
        u55
        u56
        u57
        u58
        u59
        u5a
        u61
        u62
        u63
        u64
        u65
        u66
        u67
        u68
        u69
        u6a
        u6b
        u6c
        u6d
        u6e
        u6f
        u70
        u71
        u72
        u73
        u74
        u75
        u76
        u77
        u78
        u79
        u7a
        ua8
        uaf
        ud7
        uf7
        u20
        u21
        u22
        u23
        u24
        u25
        u26
        u27
        u28
        u29
        u2a
        u2b
        u2c
        u2d
        u2e
        u2f
        u3a
        u3b
        u3c
        u3d
        u3e
        u3f
        u40
        u5b
        u5d
        u5e
        u5f
        u60
        u7b
        u7c
        u7d
        u7e
        u5c""".split("\n")))
    
    def __init__(self, filename: str):
        with open(filename, "r") as f:
            yaml_object = load(f.read(), Loader=Loader)
        
        yaml_object = yaml_object["handwriting"]

        global_config = yaml_object["global_transforms"]

        self._config_dict = {}

        for char in self.CHARACTER_CODES:
            if char not in yaml_object["character_transforms"]:
                self._config_dict[char] = global_config
            else:
                custom_key = yaml_object["character_transforms"][char]
                self._config_dict[char] = self._merge_configs(global_config, yaml_object["custom_transforms"][custom_key])

    def _merge_configs(self, base: dict, overlap: dict) -> dict:
        assert all(x in base.keys() for x in overlap.keys())

        new_config = {}
        for key in base.keys():
            new_config[key] = overlap[key] if key in overlap else base[key]

        return new_config

    def _get_attribute(self, character_unicode: str, transform: str) -> int:
        config = self._config_dict[character_unicode][transform]
        value = int(random.normalvariate(config["mean"], config["var"]))
        value = max(value, config["min"])
        value = min(value, config["max"])

        return value

    def get_rotation(self, character_unicode: str) -> int:
        return self._get_attribute(character_unicode, "rotation")
            
    def get_translation_y(self, character_unicode: str) -> int:
        return self._get_attribute(character_unicode, "translation_y")
    
    def get_translation_x(self, character_unicode: str) -> int:
        return self._get_attribute(character_unicode, "translation_x")

    def get_height(self, character_unicode: str) -> int:
        return self._get_attribute(character_unicode, "height")
    
    def get_width(self, character_unicode: str) -> int:
        return self._get_attribute(character_unicode, "width")
    
    def get_thickness(self) -> int:
        return self._get_attribute("u20", "thickness")

if __name__ == "__main__":
    my_config = Config("example_individual_config.yml")

    print(my_config.get_stitch("u32"))
    print(my_config.get_stitch("u27"))
    print(my_config.get_stitch("u2b"))

    print(my_config.get_height("u27"))
    print(my_config.get_height("u27"))
    print(my_config.get_height("u27"))
    print(my_config.get_height("u27"))
    print(my_config.get_height("u27"))
    