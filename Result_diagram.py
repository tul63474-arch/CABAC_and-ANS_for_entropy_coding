print("\n" + "="*60)
print("GENERATING CORRECTED LINE DIAGRAMS & OUTPUT VIDEOS")
print("="*60)

# Calculate entropy if not already defined
try:
    entropy
except NameError:
    hist = np.bincount(symbols, minlength=11)
    probs = hist[hist > 0] / len(symbols)
    entropy = 0
    for p in probs:
        entropy -= p * math.log2(p)
    print(f"Calculated entropy: {entropy:.4f} bits/symbol")

# ============================================
# PART 1: GENERATE OUTPUT VIDEOS
# ============================================
print("\n📹 Generating Output Videos...")

if len(frames_rgb) > 0:
    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    
    # CABAC Output Video
    cabac_video = 'cabac_output.mp4'
    out1 = cv2.VideoWriter(cabac_video, fourcc, fps, (w, h))
    for frame in frames_rgb[:100]:
        frame_uint8 = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        out1.write(frame_bgr)
    out1.release()
    print(f"  ✓ {cabac_video}")
    display(Video(cabac_video, embed=True, width=480))
    files.download(cabac_video)
    
    # ANS Output Video
    ans_video = 'ans_output.mp4'
    out2 = cv2.VideoWriter(ans_video, fourcc, fps, (w, h))
    for frame in frames_rgb[:100]:
        frame_uint8 = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        out2.write(frame_bgr)
    out2.release()
    print(f"  ✓ {ans_video}")
    display(Video(ans_video, embed=True, width=480))
    files.download(ans_video)

# ============================================
# PART 2: CORRECTED LINE DIAGRAMS
# ============================================
print("\n Generating Corrected Line Diagrams...")

# Set clean style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Create figure with 4 line diagram subplots
fig = plt.figure(figsize=(14, 10))
fig.suptitle('Entropy Coding Analysis: CABAC vs ANS', fontsize=14, fontweight='bold', y=0.98)

# ============================================
# PLOT 1: CODING EFFICIENCY OVER TIME
# ============================================
ax1 = plt.subplot(2, 2, 1)

# Limit points
n = min(500, len(cabac.symbol_bits), len(ans.symbol_bits))

# Plot both lines directly
ax1.plot(range(n), cabac.symbol_bits[:n], 'b-', linewidth=2, label='CABAC')
ax1.plot(range(n), ans.symbol_bits[:n], 'r-', linewidth=2, label='ANS')

ax1.set_xlabel('Symbol Index', fontsize=11)
ax1.set_ylabel('Bits per Symbol', fontsize=11)
ax1.set_title('(a) Coding Efficiency Over Time', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add mean lines
ax1.axhline(y=np.mean(cabac.symbol_bits[:n]), color='blue', linestyle=':', alpha=0.7)
ax1.axhline(y=np.mean(ans.symbol_bits[:n]), color='red', linestyle=':', alpha=0.7)

# ============================================
# PLOT 2: CUMULATIVE BIT USAGE (WITH THEORETICAL LINE)
# ============================================
ax2 = plt.subplot(2, 2, 2)

num_points = min(1000, len(cabac.symbol_bits))
cabac_cumulative = np.cumsum(cabac.symbol_bits[:num_points])
ans_cumulative = np.cumsum(ans.symbol_bits[:num_points])

ax2.plot(range(num_points), cabac_cumulative, label='CABAC', linewidth=2, color='#3498db')
ax2.plot(range(num_points), ans_cumulative, label='ANS', linewidth=2, color='#e74c3c')

# Add theoretical entropy line (FIXED - now included)
theoretical_cumulative = np.cumsum([entropy] * num_points)
ax2.plot(range(num_points), theoretical_cumulative, label=f'Shannon Entropy ({entropy:.3f} bits/symbol)', 
         linewidth=1.5, color='gray', linestyle='--', alpha=0.8)

ax2.set_xlabel('Symbol Index', fontsize=11)
ax2.set_ylabel('Cumulative Bits', fontsize=11)
ax2.set_title('(b) Cumulative Bit Usage', fontsize=12, fontweight='bold')
# Move legend to bottom right to avoid covering the lines
ax2.legend(loc='lower right', frameon=True, fancybox=True, fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')

# ============================================
# PLOT 3: BIT DISTRIBUTION
# ============================================
ax3 = plt.subplot(2, 2, 3)

# Create histograms as line plots
max_bits = max(max(cabac.symbol_bits), max(ans.symbol_bits))
bins = np.linspace(0, max_bits, 40)
cabac_hist, bin_edges = np.histogram(cabac.symbol_bits, bins=bins, density=True)
ans_hist, _ = np.histogram(ans.symbol_bits, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

ax3.plot(bin_centers, cabac_hist, label='CABAC', linewidth=2, color='#3498db', marker='o', markersize=4)
ax3.plot(bin_centers, ans_hist, label='ANS', linewidth=2, color='#e74c3c', marker='s', markersize=4)
ax3.set_xlabel('Bits per Symbol', fontsize=11)
ax3.set_ylabel('Probability Density', fontsize=11)
ax3.set_title('(c) Bit Distribution', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', frameon=True, fancybox=True)
ax3.grid(True, alpha=0.3, linestyle='--')

# Add vertical lines for means
ax3.axvline(x=mean_cabac, color='#3498db', linestyle='--', alpha=0.5, linewidth=1)
ax3.axvline(x=mean_ans, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)


# ============================================
# PLOT 4: DIRECT METRICS COMPARISON
# ============================================
ax4 = plt.subplot(2, 2, 4)

# Use ACTUAL values
metrics = ['Compression\nRatio', 'Bits per\nSymbol', 'Latency\n(ms)']
cabac_actual = [cabac_cr, cabac_bits/len(symbols), cabac.encoding_time*1000]
ans_actual = [ans_cr, ans_bits/len(symbols), ans.encoding_time*1000]

x_pos = range(len(metrics))

# Plot lines with markers
ax4.plot(x_pos, cabac_actual, label='CABAC', linewidth=2, color='#3498db', 
         marker='o', markersize=8, linestyle='-')
ax4.plot(x_pos, ans_actual, label='ANS', linewidth=2, color='#e74c3c', 
         marker='s', markersize=8, linestyle='-')

ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics)
ax4.set_ylabel('Value', fontsize=11)
ax4.set_title('(d) Performance Metrics Comparison', fontsize=12, fontweight='bold')
# Move legend to lower left to avoid covering
ax4.legend(loc='lower left', frameon=True, fancybox=True)
ax4.grid(True, alpha=0.3, linestyle='--')

# Add value labels with careful positioning to avoid overlap
for i, (c_val, a_val) in enumerate(zip(cabac_actual, ans_actual)):
    if i == 0:  # Compression ratio
        c_label = f'{c_val:.2f}x'
        a_label = f'{a_val:.2f}x'
        c_offset = 10
        a_offset = -15
    elif i == 1:  # Bits per symbol
        c_label = f'{c_val:.3f}'
        a_label = f'{a_val:.3f}'
        c_offset = 10
        a_offset = -15
    else:  # Latency
        c_label = f'{c_val:.1f}ms'
        a_label = f'{a_val:.1f}ms'
        c_offset = 10
        a_offset = -15
    
    ax4.annotate(c_label, (i, c_val), textcoords="offset points", 
                 xytext=(0, c_offset), ha='center', fontsize=9, color='#3498db', fontweight='bold')
    ax4.annotate(a_label, (i, a_val), textcoords="offset points", 
                 xytext=(0, a_offset), ha='center', fontsize=9, color='#e74c3c', fontweight='bold')

plt.tight_layout()

# Save diagrams
plt.savefig('entropy_analysis_lines.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('entropy_analysis_lines.pdf', format='pdf', bbox_inches='tight')
plt.show()

files.download('entropy_analysis_lines.png')
files.download('entropy_analysis_lines.pdf')
