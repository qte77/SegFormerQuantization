def print_model_sizes(models):
    for m in models.keys():
        print(
            f"{m} size {models[m].get_memory_footprint()*1.0e-6:.2f} MB"
            f" on {models[m].device}"
        )

