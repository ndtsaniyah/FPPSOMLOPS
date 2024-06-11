#!/bin/sh

set -o errexit
set -o nounset


/usr/local/bin/uwsgi --http 0.0.0.0:5000 \
    --plugins python3 \
    --wsgi run:app

echo ""
echo "$@"
