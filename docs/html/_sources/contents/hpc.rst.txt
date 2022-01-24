Pykanto and High Performance Computing 
======================================

parameters
----------


`Slurm <https://slurm.schedmd.com/documentation.html>`_ is arguably the most
popular job scheduler used in compute clusters \... so instructions and
resources here.

The following is adapted from `Peng Zhenghao
<https://github.com/pengzhenghao/use-ray-with-slurm>`_

1. Fetches the list of computing nodes and their IP addresses. 

2. Launches a head ray process in one of the node, and get the address of the
head node. 

3. Launches ray processes in (n-1) worker nodes and connects them to the head
node by providing the head node address. 

4. Submits the user specified task to ray.


Add this to the top of your script, after imports:

.. code-block:: python
   :linenos:

    redis_password = sys.argv[1]
    ray.init(address=os.environ["ip_head"], _redis_password=redis_password)
    print(ray.cluster_resources())