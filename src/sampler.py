import random
def negative_sample(user_items, num_items, max_tries=500):
    """
    We are sampling an item randomly that  user has not interacted with
     it would return RuntimeError if a valid negative is not found within maximum try
    """
    for _ in range(max_tries):
        neg = random.randint(1, num_items)
        if neg not in user_items:
            return neg
    raise RuntimeError(
        f"Could not sample a negative item in {max_tries} tries. "
        f"user_items size={len(user_items)}, num_items={num_items}."
    )
