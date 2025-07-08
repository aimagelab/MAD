# Formatting is name --> gpu_id --> region and prompts


regions_settings = {
    'first_map': {
        0: {
            'mask_paths': ["/home/fquattrini/MAD/masks/rocks.png", "/home/fquattrini/MAD/masks/bonfire.png"],
            'fg_prompts': ["Rocks on the beach", "A bonfire with few logs"],
            'fg_negative': ["artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image"] * 2,
            'bg_prompt': "A sandy flat beach",
            'bg_negative': "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image"
        },
        1: {
            'mask_paths': ["/home/fquattrini/MAD/masks/segments_1.png", "/home/fquattrini/MAD/masks/segments_2.png"],
            'fg_prompts': ["A dune", "An oasis"],
            'fg_negative': ["artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image"] * 2,
            'bg_prompt': "A photo of a desert",
            'bg_negative': "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image"
        },
        2: {
            'mask_paths': ["/home/fquattrini/MAD/masks/mouse.png", "/home/fquattrini/MAD/masks/tree.png"],
            'fg_prompts': ["A basketball ball", "A lego tower"],
            'fg_negative': [""] * 2,
            'bg_prompt': "A brick wall",
            'bg_negative': ""
        }
    }
}
