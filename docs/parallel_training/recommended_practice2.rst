Recommended Practice(no_wait mode)
---------------------

| This tutorial shows how to use ``@parl.remote_class`` to implement parallel computation **without** multithreading.

| In the previous tutorial, we implemented parallel computation through decorator and multithreading. PARL actually provides a more compact parallel computation mode without manually creating threads. By passing the argument ``wait=false`` to the decoratior, we can run tasks in parallel in a simpler way. The program will not be blocked while calling the functions of the decroated class. Instead, it will return a ``future_object`` immediately, and users can obtain the result in the future by calling ``future_object.get()``.

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

    1. We add ``wait=False`` in the remote decorator so that the program will not be blocked.

    2. After actors start running, calling ``job.get()`` will block the main program until the job is finished. The return of ``jog.get()`` is the same as calling the original function.


