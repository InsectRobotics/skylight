# Find path to the current file
cd "$( dirname "${BASH_SOURCE[0]}" )"
DIR="$( pwd )"/"../../skylight"

echo "The following path is going to be added to the PYTHONPATH env var: "
echo "---> " ${DIR}
#echo "This path should point to the root of the project "

# Append PATH env var with the path to the root of the project
export PYTHONPATH=${DIR}:$PYTHONPATH
#echo ${PYTHONPATH}
