import pyfocusr


def test_fake_1(timing=False):
    if timing is True:
        tic = time.time()

    for i in range(10):
        assert i == i

    if timing is True:
        toc = time.time()
        print("Test took: {}s".format(toc - tic))


if __name__ == "__main__":
    import time

    test_fake_1(timing=True)
