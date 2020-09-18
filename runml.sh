#!/bin/sh

docker stop dobro-ml

docker run -di --rm --name dobro-ml -v /`pwd`:/usr/local/proj -p 4444:4444 fogugy/dobro-ml
docker exec -di dobro-ml sh  -c "cd /usr/local/proj && python3 server.py"