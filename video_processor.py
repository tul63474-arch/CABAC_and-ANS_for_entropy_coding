class VideoProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def read_yuv_frame(self, f, width, height):
        y_size = width * height
        uv_size = y_size // 4
        y_data = f.read(y_size)
        if len(y_data) < y_size:
            return None
        u_data = f.read(uv_size)
        v_data = f.read(uv_size)
        if len(u_data) < uv_size or len(v_data) < uv_size:
            return None
        y = np.frombuffer(y_data, dtype=np.uint8).reshape(height, width)
        u = np.frombuffer(u_data, dtype=np.uint8).reshape(height//2, width//2)
        v = np.frombuffer(v_data, dtype=np.uint8).reshape(height//2, width//2)
        u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
        y = y.astype(np.float32) / 255.0
        u = u.astype(np.float32) / 255.0 - 0.5
        v = v.astype(np.float32) / 255.0 - 0.5
        r = y + 1.402 * v
        g = y - 0.344 * u - 0.714 * v
        b = y + 1.772 * u
        rgb = np.stack([r, g, b], axis=2)
        rgb = np.clip(rgb, 0, 1)
        return rgb
    
    def read_y4m_frame(self, f, width, height):
        line = f.readline()
        while line and not line.startswith(b'FRAME'):
            if not line:
                return None
            line = f.readline()
        if not line:
            return None
        y_size = width * height
        uv_size = y_size // 4
        y_data = f.read(y_size)
        if len(y_data) < y_size:
            return None
        u_data = f.read(uv_size)
        v_data = f.read(uv_size)
        if len(u_data) < uv_size or len(v_data) < uv_size:
            return None
        y = np.frombuffer(y_data, dtype=np.uint8).reshape(height, width)
        u = np.frombuffer(u_data, dtype=np.uint8).reshape(height//2, width//2)
        v = np.frombuffer(v_data, dtype=np.uint8).reshape(height//2, width//2)
        u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
        y = y.astype(np.float32) / 255.0
        u = u.astype(np.float32) / 255.0 - 0.5
        v = v.astype(np.float32) / 255.0 - 0.5
        r = y + 1.402 * v
        g = y - 0.344 * u - 0.714 * v
        b = y + 1.772 * u
        rgb = np.stack([r, g, b], axis=2)
        rgb = np.clip(rgb, 0, 1)
        return rgb
    
    def extract_features(self, frame):
        h, w = frame.shape[:2]
        symbols = []
        contexts = []
        for i in range(0, h, 16):
            for j in range(0, w, 16):
                if i+16 <= h and j+16 <= w:
                    block = frame[i:i+16, j:j+16, 0]
                    residual = np.std(block)
                    symbol = int(residual * 10)
                    symbol = min(10, max(0, symbol))
                    symbols.append(symbol)
                    if residual < 0.1:
                        contexts.append(0)
                    elif residual < 0.3:
                        contexts.append(1)
                    else:
                        contexts.append(2)
        return symbols, contexts
    
    def process(self, max_frames, width, height):
        print(f"Processing video file...")
        all_symbols = []
        all_contexts = []
        frames_rgb = []
        frames_loaded = 0
        is_y4m = self.file_path.endswith('.y4m')
        
        with open(self.file_path, 'rb') as f:
            if is_y4m:
                f.readline()
            for frame_num in range(max_frames):
                if is_y4m:
                    frame = self.read_y4m_frame(f, width, height)
                else:
                    frame = self.read_yuv_frame(f, width, height)
                if frame is None:
                    break
                frames_rgb.append(frame)
                symbols, contexts = self.extract_features(frame)
                all_symbols.extend(symbols)
                all_contexts.extend(contexts)
                frames_loaded += 1
                if (frame_num + 1) % 10 == 0:
                    print(f"  Processed {frame_num + 1} frames...")
        
        print(f"Loaded {frames_loaded} frames")
        print(f"Generated {len(all_symbols)} symbols")
        return all_symbols, all_contexts, frames_loaded, frames_rgb

print(" Video Processor defined!")
