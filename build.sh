#!/usr/bin sh

export PING_SLEEP=30s
export WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILD_OUTPUT=$WORKDIR/build.out

touch $BUILD_OUTPUT

dump_output() {
   echo Tailing the last 500 lines of output:
   tail -500 $BUILD_OUTPUT
}
error_handler() {
  echo ERROR: An error was encountered with the build.
  dump_output
  exit 1
}
# If an error occurs, run our error handler to output a tail of the build
trap 'error_handler' ERR

bash -c "while true; do echo \$(date) - building ...; sleep $PING_SLEEP; done" &
PING_LOOP_PID=$!
# Create a virtual environment
#virtualenv -p python3 ~/venvs/bhabana
#source ~/venvs/bhabana/bin/activate

# Install all the dev dependencies
pip install -r requirements/requirements.txt >> $BUILD_OUTPUT 2>&1

# Install the Spacy models for DE and Eng
python -m spacy download en_vectors_web_lg >> $BUILD_OUTPUT 2>&1
python -m spacy download de_core_news_sm >> $BUILD_OUTPUT 2>&1
python -m spacy download en_core_web_lg >> $BUILD_OUTPUT 2>&1

# The build finished without returning an error so dump a tail of the output
dump_output

# nicely terminate the ping output loop
kill $PING_LOOP_PID