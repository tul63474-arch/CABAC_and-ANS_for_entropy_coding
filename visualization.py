!pip install gradio numpy matplotlib opencv-python-headless pandas -q
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import math
import tempfile
import os
import re
from collections import defaultdict

print("Starting application...")

# CABAC Encoder
class CABACEncoder:
    def __init__(self):
        self.prob_states = {0: 0.5, 1: 0.5, 2: 0.5}
        self.total_bits = 0
        self.bit_breakdown = defaultdict(float)
        self.encoding_time = 0
        self.symbol_bits = []
        
    def encode_sequence(self, symbols, contexts):
        start_time = time.time()
        self.symbol_bits = []
        for sym, ctx in zip(symbols, contexts):
            prob = self.prob_states[ctx]
            bits = -math.log2(prob if sym > 5 else 1-prob)
            self.total_bits += bits
            self.symbol_bits.append(bits)
            self.bit_breakdown[ctx] += bits
            
            if sym > 5:
                self.prob_states[ctx] += 0.1 * (1 - self.prob_states[ctx])
            else:
                self.prob_states[ctx] -= 0.1 * self.prob_states[ctx]
            self.prob_states[ctx] = np.clip(self.prob_states[ctx], 0.01, 0.99)
        
        self.encoding_time = time.time() - start_time
        return self.total_bits

# ANS Encoder
class ANSEncoder:
    def __init__(self, scale=4096):
        self.scale = scale
        self.total_bits = 0
        self.encoding_time = 0
        self.symbol_bits = []
        self.freq_table = None
        self.cdf_table = None
        
    def build_model(self, symbols, num_symbols=11):
        hist = np.bincount(symbols, minlength=num_symbols)
        probs = hist / len(symbols)
        freqs = np.maximum(1, (probs * self.scale).astype(int))
        freqs[-1] += self.scale - np.sum(freqs)
        self.freq_table = freqs
        self.cdf_table = np.cumsum(freqs)
        self.total_freq = self.cdf_table[-1]
        
    def encode_sequence(self, symbols):
        start_time = time.time()
        self.symbol_bits = []
        if self.freq_table is None:
            self.build_model(symbols)
        state = self.total_freq
        for sym in reversed(symbols):
            freq = self.freq_table[sym]
            cdf = self.cdf_table[sym] - freq
            q = state // freq
            r = state % freq
            state = (q << 16) + cdf + r
            bits = math.log2(state) if state > 0 else 0
            self.symbol_bits.append(bits)
        self.total_bits = math.log2(state) if state > 0 else 0
        self.encoding_time = time.time() - start_time
        return self.total_bits

# Video Processor
class VideoProcessor:
    def __init__(self, file_path, width=None, height=None):
        self.file_path = file_path
        self.width = width
        self.height = height
        
    def read_yuv_frame(self, f, width, height):
        try:
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
        except Exception as e:
            print(f"Error reading YUV frame: {e}")
            return None
    
    def read_mp4_frames(self, max_frames=50):
        frames = []
        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {self.file_path}")
            return frames
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_norm = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_norm)
            frame_count += 1
        cap.release()
        print(f"Read {len(frames)} frames from MP4")
        return frames
    
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
    
    def process(self, max_frames=50, width=352, height=288, video_format="MP4"):
        frames = []
        
        try:
            if video_format in ["YUV", "Y4M"]:
                with open(self.file_path, 'rb') as f:
                    if video_format == "Y4M":
                        header = f.readline()
                        w_match = re.search(rb'W(\d+)', header)
                        h_match = re.search(rb'H(\d+)', header)
                        if w_match and h_match:
                            width = int(w_match.group(1))
                            height = int(h_match.group(1))
                            print(f"Detected Y4M resolution: {width}x{height}")
                    
                    frame_count = 0
                    while frame_count < max_frames:
                        frame = self.read_yuv_frame(f, width, height)
                        if frame is None:
                            break
                        frames.append(frame)
                        frame_count += 1
                    print(f"Read {len(frames)} frames from {video_format} file")
            else:
                frames = self.read_mp4_frames(max_frames)
                
        except Exception as e:
            print(f"Error processing video: {e}")
            return [], [], []
        
        if len(frames) == 0:
            print("No frames were extracted!")
            return [], [], []
        
        all_symbols = []
        all_contexts = []
        for i, frame in enumerate(frames):
            symbols, contexts = self.extract_features(frame)
            all_symbols.extend(symbols)
            all_contexts.extend(contexts)
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(frames)} frames")
        
        print(f"Extracted {len(all_symbols)} symbols")
        return all_symbols, all_contexts, frames

# Analysis function
def analyze_video(video_file, max_frames, width, height, video_format):
    print(f"\n=== Starting Analysis ===")
    print(f"Format: {video_format}")
    print(f"Max frames: {max_frames}")
    print(f"Resolution: {width}x{height}")
    
    if video_file is None:
        return "Error: Please upload a video file", None, None
    
    # Handle different file types from Gradio
    try:
        # Gradio File component returns an object with 'name' attribute
        if hasattr(video_file, 'name'):
            file_path = video_file.name
            print(f"File path: {file_path}")
        else:
            return "Error: Invalid file object", None, None
            
    except Exception as e:
        return f"Error accessing file: {str(e)}", None, None
    
    # Process video
    try:
        processor = VideoProcessor(file_path, width, height)
        symbols, contexts, frames = processor.process(max_frames, width, height, video_format)
        
        if len(symbols) == 0:
            return "Error: No symbols extracted. Check your video format and resolution.", None, None
        
        print(f"Processing {len(symbols)} symbols...")
        
        # Run CABAC
        cabac = CABACEncoder()
        cabac_bits = cabac.encode_sequence(symbols, contexts)
        
        # Run ANS
        ans = ANSEncoder()
        ans_bits = ans.encode_sequence(symbols)
        
        # Calculate metrics
        original_bits = len(symbols) * 8
        cabac_cr = min(original_bits / cabac_bits, 3.0)
        ans_cr = min(original_bits / ans_bits, 3.0)
        
        duration = max_frames / 30
        cabac_bitrate = (cabac_bits / duration) / 1_000_000
        ans_bitrate = (ans_bits / duration) / 1_000_000
        
        # Entropy
        hist = np.bincount(symbols, minlength=11)
        probs = hist[hist > 0] / len(symbols)
        entropy = -sum(p * math.log2(p) for p in probs)
        
        total_cabac_bits = sum(cabac.bit_breakdown.values())
        
        # Create results text
        results_text = f"""
========================================
CABAC vs ANS - ANALYSIS RESULTS
========================================

Input File: {os.path.basename(file_path)}
Format: {video_format}
Resolution: {width}x{height}
Frames processed: {len(frames)}
Symbols processed: {len(symbols):,}

========================================
QUANTITATIVE METRICS
========================================
Metric                    CABAC           ANS
--------------------------------------------------
Compression Ratio         {cabac_cr:.2f}x             {ans_cr:.2f}x
Bitrate (Mbps)           {cabac_bitrate:.4f}          {ans_bitrate:.4f}
Latency (ms)             {cabac.encoding_time*1000:.1f}          {ans.encoding_time*1000:.1f}
Bits per Symbol          {cabac_bits/len(symbols):.3f}          {ans_bits/len(symbols):.3f}
Total Bits               {cabac_bits:.0f}          {ans_bits:.0f}

Theoretical Shannon Entropy: {entropy:.4f} bits/symbol

========================================
CABAC BIT USAGE BREAKDOWN
========================================
Smooth Areas:    {cabac.bit_breakdown.get(0, 0):.0f} bits ({(cabac.bit_breakdown.get(0, 0)/total_cabac_bits*100):.1f}%)
Texture Regions: {cabac.bit_breakdown.get(1, 0):.0f} bits ({(cabac.bit_breakdown.get(1, 0)/total_cabac_bits*100):.1f}%)
Edges/Motion:    {cabac.bit_breakdown.get(2, 0):.0f} bits ({(cabac.bit_breakdown.get(2, 0)/total_cabac_bits*100):.1f}%)

========================================
PERFORMANCE SUMMARY
========================================
Best Compression: {'CABAC' if cabac_cr > ans_cr else 'ANS'} ({abs((cabac_cr/ans_cr-1)*100) if cabac_cr > ans_cr else abs((ans_cr/cabac_cr-1)*100):.1f}% better)
Fastest Encoding: {'ANS' if ans.encoding_time < cabac.encoding_time else 'CABAC'} ({cabac.encoding_time/ans.encoding_time:.1f}x faster)
========================================
"""
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('CABAC vs ANS - Entropy Coding Analysis', fontsize=14, fontweight='bold')
        
        # Plot 1: CABAC Bit Breakdown
        ax1 = axes[0, 0]
        names = ['Smooth', 'Texture', 'Edge']
        values = [cabac.bit_breakdown.get(0, 0), cabac.bit_breakdown.get(1, 0), cabac.bit_breakdown.get(2, 0)]
        bars = ax1.bar(names, values, color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black')
        ax1.set_title('CABAC Bit Usage by Context')
        ax1.set_ylabel('Bits Used')
        for bar, val in zip(bars, values):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.0f}', ha='center', va='bottom')
        
        # Plot 2: Compression Ratio
        ax2 = axes[0, 1]
        bars = ax2.bar(['CABAC', 'ANS'], [cabac_cr, ans_cr], color=['#3498db', '#e74c3c'], edgecolor='black')
        ax2.set_title('Compression Ratio')
        ax2.set_ylabel('Ratio (x)')
        for bar, val in zip(bars, [cabac_cr, ans_cr]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}x', ha='center', va='bottom')
        
        # Plot 3: Latency and Bitrate
        ax3 = axes[1, 0]
        x = np.arange(2)
        width_bar = 0.35
        ax3.bar(x - width_bar/2, [cabac.encoding_time*1000, ans.encoding_time*1000], width_bar, label='Latency (ms)', color='#3498db', edgecolor='black')
        ax3.bar(x + width_bar/2, [cabac_bitrate, ans_bitrate], width_bar, label='Bitrate (Mbps)', color='#e74c3c', edgecolor='black')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['CABAC', 'ANS'])
        ax3.set_title('Performance Metrics')
        ax3.legend()
        
        # Plot 4: Coding Efficiency
        ax4 = axes[1, 1]
        window = min(50, len(cabac.symbol_bits) // 20)
        if window > 1 and len(cabac.symbol_bits) > 0:
            cabac_smooth = np.convolve(cabac.symbol_bits, np.ones(window)/window, mode='valid')
            ans_smooth = np.convolve(ans.symbol_bits, np.ones(window)/window, mode='valid')
            if len(cabac_smooth) > 0:
                ax4.plot(range(len(cabac_smooth)), cabac_smooth, label='CABAC', color='#3498db', linewidth=2)
                ax4.plot(range(len(ans_smooth)), ans_smooth, label='ANS', color='#e74c3c', linewidth=2)
        ax4.set_xlabel('Symbol Window')
        ax4.set_ylabel('Bits per Symbol')
        ax4.set_title('Coding Efficiency Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=entropy, color='gray', linestyle='--', label=f'Entropy: {entropy:.3f}')
        ax4.legend()
        
        plt.tight_layout()
        
        # Create output video
        output_video_path = None
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            output_video_path = '/tmp/output_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
            for frame in frames[:min(100, len(frames))]:
                frame_uint8 = (frame * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"Output video saved: {output_video_path}")
        
        print("Analysis completed successfully!")
        return results_text, fig, output_video_path
        
    except Exception as e:
        import traceback
        error_msg = f"Error during analysis: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, None, None

# Create Gradio Interface
print("Creating Gradio interface...")

with gr.Blocks(title="CABAC vs ANS Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CABAC vs ANS - Entropy Coding Analysis")
    gr.Markdown("Comparative analysis for video compression. Supports MP4, YUV, and Y4M formats.")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(label="Upload Video File", file_types=[".mp4", ".avi", ".mov", ".yuv", ".y4m"])
            format_radio = gr.Radio(choices=["MP4", "YUV", "Y4M"], label="Video Format", value="MP4")
            
            with gr.Row():
                width_input = gr.Number(label="Width (for YUV/Y4M)", value=352, visible=True)
                height_input = gr.Number(label="Height (for YUV/Y4M)", value=288, visible=True)
            
            frame_slider = gr.Slider(minimum=10, maximum=100, value=50, step=10, label="Maximum Frames to Process")
            analyze_btn = gr.Button("Run Analysis", variant="primary")
        
        with gr.Column(scale=1):
            video_output = gr.Video(label="Output Video")
    
    with gr.Row():
        with gr.Column(scale=1):
            results_output = gr.Textbox(label="Analysis Results", lines=25)
        with gr.Column(scale=1):
            plot_output = gr.Plot(label="Visualization")
    
    # Show/hide resolution inputs based on format
    def update_resolution_visibility(format_type):
        if format_type in ["YUV", "Y4M"]:
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)
    
    format_radio.change(
        fn=update_resolution_visibility,
        inputs=[format_radio],
        outputs=[width_input, height_input]
    )
    
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, frame_slider, width_input, height_input, format_radio],
        outputs=[results_output, plot_output, video_output]
    )
    
    gr.Markdown("""
    ---
    ### Instructions
    1. **Select your video format** (MP4, YUV, or Y4M)
    2. **Upload your video file**
    3. **For YUV files**: Enter correct width and height (e.g., 352x288 for CIF)
    4. **For Y4M files**: Resolution auto-detected
    5. **Click "Run Analysis"**
    
    ### About
    - **CABAC**: Context-Adaptive Binary Arithmetic Coding (H.264/HEVC)
    - **ANS**: Asymmetric Numeral Systems (faster alternative)
    """)

print("Launching Gradio interface...")
demo.launch(share=True, debug=False)
