# xparl security and upgrade guide

xparl executes Python code supplied by trusted cluster members. Possession of the cluster secret grants permission to execute code on workers. Use dedicated, least-privileged accounts and a private network; never expose cluster ports to the public Internet.

## Required authentication

Set `XPARL_AUTH_TOKEN` to the same cryptographically random secret on the master, workers, and clients **before starting their processes**. The secret must contain at least 32 bytes. Generate a fresh secret locally with `python -c 'import secrets; print(secrets.token_hex(32))'`, then distribute it through your secret management system. Do not put the secret in source control, command-line arguments, screenshots, or logs.

Every xparl ZeroMQ connection now requires CURVE authentication and encryption. The server accepts only the client public key derived from the cluster secret; knowing the server public key does not authorize a client. Startup fails when the secret is absent, too short, or CURVE is unavailable. There is no unauthenticated compatibility mode. Restart all nodes and clients together after upgrading or rotating the secret.

Master control messages, worker/job metadata, client/worker status, and monitor status use validated, data-only JSON. Pickle metadata from old versions is rejected. Python classes, arguments, return values, and source files on the authenticated execution channel continue to support Python serialization; only trusted code and trusted secret holders belong in a cluster.

## Bind addresses and monitoring

The master control socket, HTTP monitor, and HTTP log server bind to `127.0.0.1` by default. For a multi-machine private cluster, explicitly set `XPARL_BIND_HOST` to the appropriate private interface address on each node, or `0.0.0.0` behind a firewall that permits only trusted cluster nodes. Worker and job ZeroMQ sockets also require authentication, including dynamically allocated ports. Heartbeat gRPC ports still require network isolation.

All HTTP monitor and log routes require HTTP Basic authentication: username `xparl`, password the cluster secret. The browser prompts for credentials. Use an SSH tunnel or HTTPS reverse proxy for remote HTTP access because HTTP Basic authentication does not encrypt the password. Do not share the cluster secret with users who should only view monitoring data: it also authorizes execution on workers.

## Upgrading affected installations

The fix is in the `develop` source branch. Existing PyPI packages and earlier tags are not changed by a source merge. Install the patched source commit on all nodes, or upgrade to a subsequently published release containing it. This JSON/CURVE protocol is incompatible with older xparl processes. Stop the old cluster, configure the secret and private bind addresses, upgrade every node/client, and then restart.

For an installation that cannot yet upgrade, restrict the master, monitor, log, worker/job, and heartbeat ports to trusted hosts through firewalls or security groups. The reported ports 8010/8137/8200 are examples; check actual configuration and dynamic ports. Remove public exposure immediately. If an old installation was exposed, investigate its host and logs; applying a source patch does not establish whether it was previously compromised.

## Regression checks

Run `python -m unittest discover -s parl/remote/tests -p control_security_test.py -v` in a PARL environment. The suite checks plaintext clients, wrong cluster secrets, unrelated CURVE keys, malicious pickle metadata in all four master branches, invalid JSON and message frames, valid status/monitor messages, HTTP authentication, and fail-closed configuration. The `xparl security` GitHub Actions workflow runs these checks with Python 3.9 and 3.10 and the existing pyzmq 22.3.0 pin.
