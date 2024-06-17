# sets up the environment for the Vollo SDK

if [ -n "$BASH_SOURCE" ]; then
  script_location=$BASH_SOURCE
elif [ -n "$ZSH_VERSION" ]; then
  setopt function_argzero
  script_location=$0
else
  echo 1>&2 "Unsupported shell. Please use bash or zsh or manually set VOLLO_SDK, PATH, and LD_LIBRARY_PATH"
  exit 2
fi

(return 0 2>/dev/null) || { >&2 echo -e "This setup script must be sourced. Run:\n  source $script_location"; exit 1; }

export VOLLO_SDK=$(cd -- "$( dirname -- "$script_location" )" &> /dev/null && pwd)
export PATH=$VOLLO_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VOLLO_SDK/lib${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH
>&2 echo "Setup ran for vollo-sdk-$(cat $VOLLO_SDK/VERSION)"
>&2 echo "VOLLO_SDK set to: $VOLLO_SDK"
>&2 echo "Added to PATH: $VOLLO_SDK/bin"
>&2 echo "Added to LD_LIBRARY_PATH: $VOLLO_SDK/lib"

