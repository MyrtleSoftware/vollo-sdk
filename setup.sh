# sets up the environment for the VOLLO SDK

(return 0 2>/dev/null) || { >&2 echo -e "This setup script must be sourced. Run:\n  source ${BASH_SOURCE[0]}"; exit 1; }

export VOLLO_SDK=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
export LD_LIBRARY_PATH=$VOLLO_SDK/lib${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH
>&2 echo "setup ran for vollo-sdk-$(cat $VOLLO_SDK/VERSION)"

