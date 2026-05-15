#!/bin/bash

# Define variables
ARCH=$1
CODESERVER_VERSION=$2
PORT=${3:-9687} # Default port is 9687 if not provided
COS_URL=http://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/qs/code-server
CODESERVER_DOWNLOAD="${COS_URL}/releases/download/v${CODESERVER_VERSION}/code-server-${CODESERVER_VERSION}-linux-${ARCH}.tar.gz"
INSTALL_DIR="$HOME/.local/lib/code-server-${CODESERVER_VERSION}"
SYMLINK="$HOME/.local/bin/code-server"

# Following https://coder.com/docs/code-server/latest/install#standalone-releases
mkdir -p ~/.local/lib ~/.local/bin

# Clean up any previous installation
if [ -d "$INSTALL_DIR" ]; then
    echo "Removing previous code-server installation: $INSTALL_DIR"
    rm -rf "$INSTALL_DIR"
fi

# Remove existing symbolic link if it exists
if [ -L "$SYMLINK" ]; then
    echo "Removing existing symbolic link: $SYMLINK"
    rm "$SYMLINK"
fi

# Download and extract code-server using wget
echo "Downloading from: $CODESERVER_DOWNLOAD"
wget -O - "$CODESERVER_DOWNLOAD" | tar -C ~/.local/lib -xz

# Move and create symbolic link for code-server
mv ~/.local/lib/code-server-${CODESERVER_VERSION}-linux-${ARCH} "$INSTALL_DIR"
ln -s "$INSTALL_DIR/bin/code-server" "$SYMLINK"

# Ensure code-server binary exists
if [ ! -f "$SYMLINK" ]; then
    echo "Error: code-server binary not found at $SYMLINK"
    exit 1
fi

# Write start commands to /root/run-services.sh
RUN_SCRIPT="/root/run-services.sh"
echo "#!/bin/bash" > "$RUN_SCRIPT"
echo "unset NODE_OPTIONS" >> "$RUN_SCRIPT"
echo "nohup \"$SYMLINK\" --bind-addr 0.0.0.0:$PORT --auth none &" >> "$RUN_SCRIPT"
chmod +x "$RUN_SCRIPT"

# Execute the commands
echo "Starting code-server on port $PORT..."
unset NODE_OPTIONS
nohup "$SYMLINK" --bind-addr 0.0.0.0:$PORT --auth none > code-server.log 2>&1 &

echo "Code-server installation complete. Code-server started on port $PORT."
