def max_difference(xs):
    print 'test'
    min_elem = xs[0]
    max_elem = xs[0]
    max_diff = -1

    for elem in xs[1:]:
        min_elem = min(elem, min_elem)
        if elem > max_elem:
            max_diff = max(max_diff, elem - min_elem)
            max_elem = elem
    	    print 'hi' 
    return max_diff
