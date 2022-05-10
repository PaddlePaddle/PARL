Recommended Practice2
---------------------

| This tutorial shows how to use ``@parl.remote_class`` to implement parallel computation **without** multithreading.


| Recall that in the previous tutorial, we introduced a multithreading style parallel computation which looks as below.

.. code-block:: python

    import threading
    import parl

    @parl.remote_class
    class A(object):
        def run(self):
            ans = 0
            for i in range(100000000):
                ans += i
    threads = []
    parl.connect("localhost:6006")
    for _ in range(5):
        a = A()
        th = threading.Thread(target=a.run)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

| Now let's look at how to implement it without ``threading``.

.. code-block:: python

    import parl

    @parl.remote_class(wait=False)
    class A(object):
        def run(self):
            ans = 0
            for i in range(100000000):
                ans += i
            return ans

    parl.connect("localhost:6006")
    actors = [A() for _ in range(5)]
    jobs = [actor.run() for actor in actors]
    returns = [job.get() for job in jobs]

    true_result = sum([i for i in range(100000000)])
    for result in returns:
        assert result == true_result

| two things to notice: 

    1. We add ``wait=False`` to enable actors execute in parallel.

    2. After actors start running, calling ``job.get()`` would block the main program until the job is finished and receive the return result.


