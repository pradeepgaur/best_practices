def mysum(iterable):
    result = 0
    for val in iterable:
        result += val
    return result

def test_new_sum_list():
    assert mysum([1, 2, 3]) == 6
    
def test_new_sum_tuple():
    assert mysum((-1, 2, 3)) == 6