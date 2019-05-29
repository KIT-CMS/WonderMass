#!/bin/bash
set -x

if [ -z "$HOME" ] ; then export HOME=$PWD ; fi # needed for setup_env in batch

curl -O https://raw.githubusercontent.com/KIT-CMS/WonderMass/master/bin/generation/setup_env.sh
source setup_env.sh

WonderMass/generation/scripts/./run.sh "$@"

exit
