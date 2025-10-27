def prime_factorization(n):
    """
    Perform prime factorization of a given integer n.
    :param n: int
    :return:
    """
    ans = []
    # Loop from 2 to n
    for i in range(2, n + 1):

        # n % i == 0 means n is divisible by i
        while n % i == 0 and n > 0:
            ans.append(i)

            # divide n by i to remove this factor
            n = n // i
    return ans

def calculate_image_size(n_pixels):
    factors = prime_factorization(n_pixels)
    width = height = 1
    while factors:
        factor = factors.pop()
        if width <= height:
            width *= factor
        else:
            height *= factor
    return width, height

def remove_extension(filename):
    """
    Remove the extension from a filename.
    :param filename: str
    :return:
    """
    return '.'.join(filename.split('.')[:-1])