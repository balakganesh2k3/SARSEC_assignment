import random
def negative_sample(user_items, num_items, max_tries=500):
    for _ in range(max_tries):
        neg = random.randint(1, num_items)
        if neg not in user_items:
            return neg
    raise RuntimeError(
        f"Could not sample a negative item in {max_tries} tries. "
        f"user_items size={len(user_items)}, num_items={num_items}."
    )
