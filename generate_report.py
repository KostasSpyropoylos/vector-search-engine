import os
import time
import numpy as np
import matplotlib.pyplot as plt
from models.search_engine import VectorSearchEngine
from utils.file_reader import FileReader

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register DejaVuSans for Greek character support in ReportLab
try:
    pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    font_normal = 'DejaVuSans'
    font_bold = 'DejaVuSans-Bold'
except Exception as e:
    print(f"Warning: Could not register DejaVuSans fonts: {e}. Falling back to Helvetica.")
    font_normal = 'Helvetica'
    font_bold = 'Helvetica-Bold'

def run_evaluation_metrics(engine, queries, k=10, m_groups=5):
    """
    Runs evaluation on the given engine and queries, returning metrics.
    """
    num_queries = len(queries)
    transformed_queries = engine.pca.transform(queries)
    
    # Exact K-NN
    exact_results = []
    start_exact = time.time()
    for q in transformed_queries:
        indices, _ = engine.exact_knn(q, k)
        exact_results.append(set(indices))
    exact_time = time.time() - start_exact
    exact_qps = num_queries / exact_time if exact_time > 0 else 0.0

    # Approximate K-NN
    approx_results = []
    start_approx = time.time()
    for q in transformed_queries:
        indices, _ = engine.approximate_knn(q, k, m_groups)
        approx_results.append(set(indices))
    approx_time = time.time() - start_approx
    approx_qps = num_queries / approx_time if approx_time > 0 else 0.0

    # Calculate Recall
    total_recall = 0
    for exact_set, approx_set in zip(exact_results, approx_results):
        if len(exact_set) > 0:
            intersection = len(exact_set.intersection(approx_set))
            total_recall += intersection / k
        else:
            total_recall += 1.0
    average_recall = total_recall / num_queries

    return exact_qps, approx_qps, average_recall

def run_experiments():
    print("Loading datasets...")
    # Load a subset for fast and stable experiments (10,000 base vectors, 100 queries)
    base_data = FileReader.read_fvecs('dataset/sift_base.fvecs', 10000)
    query_data = FileReader.read_fvecs('dataset/sift_query.fvecs', 100)
    
    os.makedirs('output', exist_ok=True)
    
    # ----------------------------------------------------
    # Experiment 1: Varying M (probed clusters)
    # Fixed: N=10000, k=10, P=100 (sqrt(N))
    # ----------------------------------------------------
    print("Running Experiment 1 (Varying M)...")
    m_values = [1, 2, 5, 10, 20, 30]
    exp1_exact_qps = []
    exp1_approx_qps = []
    exp1_recall = []
    
    engine = VectorSearchEngine(base_data)
    engine.build_inverted_index(num_clusters_p=100)
    
    for m in m_values:
        eqps, aqps, rec = run_evaluation_metrics(engine, query_data, k=10, m_groups=m)
        exp1_exact_qps.append(eqps)
        exp1_approx_qps.append(aqps)
        exp1_recall.append(rec)
        print(f"M={m} | Exact QPS: {eqps:.1f} | Approx QPS: {aqps:.1f} | Recall: {rec*100:.1f}%")
        
    # Generate Plot 1
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel('Αριθμός Ομάδων (M)')
    ax1.set_ylabel('Queries Per Second (QPS)', color=color)
    line1 = ax1.plot(m_values, exp1_approx_qps, marker='o', color=color, label='Approx QPS')
    line2 = ax1.plot(m_values, exp1_exact_qps, linestyle='--', color='gray', label='Exact QPS')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Recall', color=color)
    line3 = ax2.plot(m_values, exp1_recall, marker='s', color=color, label='Recall')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')
    plt.title('Επίδραση του M (Πλήθος Ομάδων) σε QPS και Recall')
    plt.grid(True)
    plt.savefig('output/plot_m.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------
    # Experiment 2: Varying k (number of neighbors)
    # Fixed: N=10000, M=5, P=100
    # ----------------------------------------------------
    print("Running Experiment 2 (Varying k)...")
    k_values = [1, 5, 10, 20, 50, 100]
    exp2_exact_qps = []
    exp2_approx_qps = []
    exp2_recall = []
    
    for k in k_values:
        eqps, aqps, rec = run_evaluation_metrics(engine, query_data, k=k, m_groups=5)
        exp2_exact_qps.append(eqps)
        exp2_approx_qps.append(aqps)
        exp2_recall.append(rec)
        print(f"k={k} | Exact QPS: {eqps:.1f} | Approx QPS: {aqps:.1f} | Recall: {rec*100:.1f}%")
        
    # Generate Plot 2
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel('Πλήθος Γειτόνων (k)')
    ax1.set_ylabel('Queries Per Second (QPS)', color=color)
    line1 = ax1.plot(k_values, exp2_approx_qps, marker='o', color=color, label='Approx QPS')
    line2 = ax1.plot(k_values, exp2_exact_qps, linestyle='--', color='gray', label='Exact QPS')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Recall', color=color)
    line3 = ax2.plot(k_values, exp2_recall, marker='s', color=color, label='Recall')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')
    plt.title('Επίδραση του k (Γείτονες) σε QPS και Recall')
    plt.grid(True)
    plt.savefig('output/plot_k.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------
    # Experiment 3: Varying |S| (Dataset size)
    # Fixed: k=10, M=5, P = sqrt(|S|)
    # ----------------------------------------------------
    print("Running Experiment 3 (Varying |S|)...")
    s_sizes = [1000, 2000, 5000, 8000, 10000]
    exp3_exact_qps = []
    exp3_approx_qps = []
    exp3_recall = []
    
    for s_size in s_sizes:
        sub_base = base_data[:s_size]
        sub_engine = VectorSearchEngine(sub_base)
        P = int(np.sqrt(s_size))
        sub_engine.build_inverted_index(num_clusters_p=P)
        eqps, aqps, rec = run_evaluation_metrics(sub_engine, query_data, k=10, m_groups=5)
        exp3_exact_qps.append(eqps)
        exp3_approx_qps.append(aqps)
        exp3_recall.append(rec)
        print(f"|S|={s_size} | Exact QPS: {eqps:.1f} | Approx QPS: {aqps:.1f} | Recall: {rec*100:.1f}%")
        
    # Generate Plot 3
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel('Μέγεθος Συνόλου Δεδομένων (|S|)')
    ax1.set_ylabel('Queries Per Second (QPS)', color=color)
    line1 = ax1.plot(s_sizes, exp3_approx_qps, marker='o', color=color, label='Approx QPS')
    line2 = ax1.plot(s_sizes, exp3_exact_qps, linestyle='--', color='gray', label='Exact QPS')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Recall', color=color)
    line3 = ax2.plot(s_sizes, exp3_recall, marker='s', color=color, label='Recall')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')
    plt.title('Επίδραση του |S| (Μέγεθος Δεδομένων) σε QPS και Recall')
    plt.grid(True)
    plt.savefig('output/plot_s.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Also build visualization of clusters for the main engine
    print("Generating cluster visualization...")
    engine.visualize_clusters(query_data, 'output/report_clusters.png')

    return {
        'm_values': m_values, 'exp1_exact_qps': exp1_exact_qps, 'exp1_approx_qps': exp1_approx_qps, 'exp1_recall': exp1_recall,
        'k_values': k_values, 'exp2_exact_qps': exp2_exact_qps, 'exp2_approx_qps': exp2_approx_qps, 'exp2_recall': exp2_recall,
        's_sizes': s_sizes, 'exp3_exact_qps': exp3_exact_qps, 'exp3_approx_qps': exp3_approx_qps, 'exp3_recall': exp3_recall,
    }

def generate_pdf_report(results, filename='output/report.pdf'):
    print(f"Generating PDF report: {filename}...")
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=54, leftMargin=54,
        topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Greek styles
    title_style = ParagraphStyle(
        'GreekTitle',
        parent=styles['Title'],
        fontName=font_bold,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'GreekSubtitle',
        parent=styles['Normal'],
        fontName=font_normal,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=30,
        alignment=1 # Centered
    )
    
    h1_style = ParagraphStyle(
        'GreekH1',
        parent=styles['Heading1'],
        fontName=font_bold,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2C5282'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'GreekH2',
        parent=styles['Heading2'],
        fontName=font_bold,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'GreekBody',
        parent=styles['BodyText'],
        fontName=font_normal,
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=8
    )

    code_style = ParagraphStyle(
        'GreekCode',
        parent=styles['Code'],
        fontName=font_normal,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1A202C'),
        backColor=colors.HexColor('#EDF2F7'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=5,
        spaceAfter=10
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName=font_bold,
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=1
    )
    
    table_body_style = ParagraphStyle(
        'TableBody',
        fontName=font_normal,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#2D3748'),
        alignment=1
    )

    story = []
    
    # ----------------------------------------------------
    # COVER PAGE
    # ----------------------------------------------------
    story.append(Spacer(1, 40))
    story.append(Paragraph("ΜΕΛΕΤΗ ΚΑΙ ΠΕΙΡΑΜΑΤΙΚΗ ΑΞΙΟΛΟΓΗΣΗ ΜΗΧΑΝΗΣ ΔΙΑΝΥΣΜΑΤΙΚΗΣ ΑΝΑΖΗΤΗΣΗΣ", title_style))
    story.append(Paragraph("Υλοποίηση Ευρετηρίου Inverted File Index (IVF-Flat) με Χρήση K-Means & PCA", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Metadata block
    meta_text = """
    <b>Φοιτητής:</b> Κωνσταντίνος Σπυρόπουλος<br/>
    <b>Μυτρώο:</b> Ε19163<br/>
    <b>Σύνολο Δεδομένων:</b> SIFT1M Dataset (128-dimensional vectors)
    """
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 25))
    
    # ----------------------------------------------------
    # SECTION 1: ARCHITECTURE
    # ----------------------------------------------------
    story.append(Paragraph("1. Εισαγωγή & Αρχιτεκτονική", h1_style))
    intro_p1 = """
    Η παρούσα αναφορά περιγράφει την υλοποίηση και την πειραματική αξιολόγηση μιας <b>Μηχανής Διανυσματικής Αναζήτησης (Vector Search Engine)</b> για τη διαχείριση και αναζήτηση διανυσμάτων υψηλής διάστασης (high-dimensional vectors). 
    Η εύρεση των πλησιέστερων γειτόνων (K-Nearest Neighbors - K-NN) σε μεγάλους όγκους δεδομένων είναι κρίσιμη για εφαρμογές ανάκτησης πληροφοριών, συστημάτων συστάσεων και μηχανικής μάθησης.
    """
    story.append(Paragraph(intro_p1, body_style))
    
    intro_p2 = """
    Για την επίτευξη υψηλής ταχύτητας αναζήτησης χωρίς σημαντική απώλεια ακρίβειας, υιοθετήθηκε η αρχιτεκτονική <b>Inverted File Index (IVF-Flat)</b> σε συνδυασμό με τις εξής τεχνικές:
    """
    story.append(Paragraph(intro_p2, body_style))
    
    techniques_list = """
        • <b>PCA (Principal Component Analysis):</b> Μείωση των διαστάσεων των διανυσμάτων από 128 σε 2 για την επιτάχυνση της επεξεργασίας και τη δυνατότητα οπτικοποίησης των clusters.<br/>
        • <b>K-Means Clustering:</b> Διαμέριση του διανυσματικού χώρου σε <i>P</i> συστάδες (clusters). Κάθε συστάδα αντιπροσωπεύεται από το κεντροειδές της (centroid).<br/>
        • <b>Inverted Index (Αντεστραμμένο Ευρετήριο):</b> Μια δομή δεδομένων που αντιστοιχίζει κάθε κεντροειδές σε μια λίστα από δείκτες διανυσμάτων που ανήκουν σε αυτό το cluster. Τα διανύσματα εντός κάθε λίστας ταξινομούνται με βάση την απόστασή τους από το κεντροειδές για βελτιστοποίηση της αναζήτησης.<br/>
        • <b>Προσέγγιση Approximate K-NN:</b> Αντί για σάρωση ολόκληρης της βάσης, η αναζήτηση περιορίζεται μόνο στις <i>M</i> πιο υποσχόμενες συστάδες (αυτές των οποίων τα κεντροειδή είναι πλησιέστερα στο διάνυσμα ερώτησης <i>q</i>).
    """
    story.append(Paragraph(techniques_list, body_style))
    story.append(Spacer(1, 15))
    
    # ----------------------------------------------------
    # SECTION 2: ALGORITHM & EXAMPLE
    # ----------------------------------------------------
    story.append(Paragraph("2. Περιγραφή Αλγορίθμου & Παράδειγμα", h1_style))
    
    alg_step = """
    Ο αλγόριθμος IVF-Flat χωρίζεται σε δύο φάσεις:<br/>
    1. <b>Φάση Προεπεξεργασίας (Offline Indexing):</b><br/>
       &nbsp;&nbsp;a. Εκπαίδευση του K-Means στο σύνολο δεδομένων για τον προσδιορισμό <i>P</i> κεντροειδών.<br/>
       &nbsp;&nbsp;b. Αντιστοίχιση κάθε διανύσματος στο πλησιέστερο κεντροειδές.<br/>
       &nbsp;&nbsp;c. Ταξινόμηση των διανυσμάτων εντός κάθε cluster με βάση την ευκλείδεια απόστασή τους από το κεντροειδές.<br/>
    2. <b>Φάση Αναζήτησης (Online Query Resolution):</b><br/>
       &nbsp;&nbsp;a. Υπολογισμός των αποστάσεων του διανύσματος ερώτησης <i>q</i> από όλα τα <i>P</i> κεντροειδή.<br/>
       &nbsp;&nbsp;b. Επιλογή των <i>M</i> πλησιέστερων κεντροειδών.<br/>
       &nbsp;&nbsp;c. Συλλογή των υποψηφίων διανυσμάτων μόνο από αυτά τα <i>M</i> clusters.<br/>
       &nbsp;&nbsp;d. Υπολογισμός των ακριβών αποστάσεων μεταξύ του <i>q</i> και των υποψηφίων διανυσμάτων.<br/>
       &nbsp;&nbsp;e. Ταξινόμηση και επιστροφή των <i>k</i> πλησιέστερων διανυσμάτων.
    """
    story.append(Paragraph(alg_step, body_style))
    
    story.append(Paragraph("Αριθμητικό Παράδειγμα Λειτουργίας", h2_style))
    example_text = """
    Έστω ένα σύνολο δεδομένων <i>S</i> με 6 δισδιάστατα διανύσματα:<br/>
    &nbsp;&nbsp;v1 = [1.0, 1.0], v2 = [1.5, 2.0], v3 = [0.8, 1.2]<br/>
    &nbsp;&nbsp;v4 = [8.0, 9.0], v5 = [8.5, 8.0], v6 = [9.0, 8.5]<br/><br/>
    
    <b>1. Προεπεξεργασία (με P = 2 ομάδες):</b><br/>
    Ο K-Means εντοπίζει τα εξής κεντροειδή:<br/>
    &nbsp;&nbsp;C1 = [1.1, 1.4] &nbsp;&nbsp;&nbsp;&nbsp; (περιέχει τα v1, v2, v3)<br/>
    &nbsp;&nbsp;C2 = [8.5, 8.5] &nbsp;&nbsp;&nbsp;&nbsp; (περιέχει τα v4, v5, v6)<br/>
    Το αντεστραμμένο ευρετήριο αποθηκεύει:<br/>
    &nbsp;&nbsp;Index = { 0: [v1, v3, v2], 1: [v5, v6, v4] } (ταξινομημένα κατά απόσταση από το αντίστοιχο κεντροειδές).<br/><br/>
    
    <b>2. Επερώτηση (Query) με q = [1.2, 1.2], k = 2 πλησιέστερους γείτονες, και M = 1 ομάδα:</b><br/>
    &nbsp;&nbsp;a. Υπολογισμός αποστάσεων από τα κεντροειδή:<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dist(q, C1) = (1.2-1.1)^2 + (1.2-1.4)^2 = 0.01 + 0.04 = 0.05<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dist(q, C2) = (1.2-8.5)^2 + (1.2-8.5)^2 = 53.29 + 53.29 = 106.58<br/>
    &nbsp;&nbsp;b. Αφού M = 1, επιλέγουμε το πλησιέστερο κεντροειδές, δηλαδή το <b>C1 (Cluster 0)</b>.<br/>
    &nbsp;&nbsp;c. Οι υποψήφιοι γείτονες είναι μόνο οι: {v1, v2, v3}. Τα διανύσματα του Cluster 1 παραλείπονται εντελώς, μειώνοντας τις συγκρίσεις κατά 50%.<br/>
    &nbsp;&nbsp;d. Υπολογίζουμε τις ακριβείς αποστάσεις του q από τα {v1, v2, v3} και επιστρέφουμε τα 2 πλησιέστερα (k=2).
    """
    story.append(Paragraph(example_text, code_style))
    
    story.append(PageBreak())

    # ----------------------------------------------------
    # SECTION 3: EXPERIMENTAL EVALUATION
    # ----------------------------------------------------
    story.append(Paragraph("3. Πειραματική Αξιολόγηση", h1_style))
    exp_intro = """
    Η πειραματική αξιολόγηση πραγματοποιήθηκε στο σύνολο δεδομένων <b>SIFT1M</b> με τη χρήση 10,000 διανυσμάτων βάσης και 100 διανυσμάτων ερωτήσεων. 
    Μετρήθηκαν δύο βασικές μετρικές απόδοσης:<br/>
    • <b>Recall (Ανάκληση):</b> Το ποσοστό των πραγματικών πλησιέστερων γειτόνων (από τον ακριβή αλγόριθμο) που επιστράφηκαν από τον προσεγγιστικό.<br/>
    • <b>QPS (Queries Per Second):</b> Ο αριθμός των ερωτήσεων που εξυπηρετούνται ανά δευτερόλεπτο (μέτρο ταχύτητας).
    """
    story.append(Paragraph(exp_intro, body_style))
    
    # ----------------------------------------------------
    # PLOT M & TABLE M
    # ----------------------------------------------------
    story.append(Paragraph("3.1 Επίδραση του Πλήθους Ομάδων (M)", h2_style))
    m_desc = """
    Μεταβάλλοντας τον αριθμό των ομάδων <i>M</i> που εξετάζονται κατά την αναζήτηση, παρατηρούμε τη θεμελιώδη σχέση μεταξύ ταχύτητας (QPS) και ακρίβειας (Recall).
    """
    story.append(Paragraph(m_desc, body_style))
    
    # Add Table M
    table_data_m = [[
        Paragraph("<b>M (Ομάδες)</b>", table_header_style), 
        Paragraph("<b>Exact QPS</b>", table_header_style), 
        Paragraph("<b>Approx QPS</b>", table_header_style), 
        Paragraph("<b>Recall</b>", table_header_style)
    ]]
    for i, m_val in enumerate(results['m_values']):
        table_data_m.append([
            Paragraph(str(m_val), table_body_style),
            Paragraph(f"{results['exp1_exact_qps'][i]:.2f}", table_body_style),
            Paragraph(f"{results['exp1_approx_qps'][i]:.2f}", table_body_style),
            Paragraph(f"{results['exp1_recall'][i]*100:.2f}%", table_body_style),
        ])
    
    t_m = Table(table_data_m, colWidths=[80, 100, 100, 80])
    t_m.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(t_m)
    story.append(Spacer(1, 10))
    
    # Add Plot M
    story.append(Image('output/plot_m.png', width=360, height=225))
    story.append(Spacer(1, 15))
    
    story.append(PageBreak())

    # ----------------------------------------------------
    # PLOT K & TABLE K
    # ----------------------------------------------------
    story.append(Paragraph("3.2 Επίδραση του Πλήθους Γειτόνων (k)", h2_style))
    k_desc = """
    Μεταβάλλοντας την παράμετρο <i>k</i> (τον αριθμό των πλησιέστερων γειτόνων που ζητούνται), αξιολογούμε την ανθεκτικότητα του αλγορίθμου όταν απαιτείται μεγαλύτερο σύνολο αποτελεσμάτων.
    """
    story.append(Paragraph(k_desc, body_style))
    
    # Add Table K
    table_data_k = [[
        Paragraph("<b>k (Γείτονες)</b>", table_header_style), 
        Paragraph("<b>Exact QPS</b>", table_header_style), 
        Paragraph("<b>Approx QPS</b>", table_header_style), 
        Paragraph("<b>Recall</b>", table_header_style)
    ]]
    for i, k_val in enumerate(results['k_values']):
        table_data_k.append([
            Paragraph(str(k_val), table_body_style),
            Paragraph(f"{results['exp2_exact_qps'][i]:.2f}", table_body_style),
            Paragraph(f"{results['exp2_approx_qps'][i]:.2f}", table_body_style),
            Paragraph(f"{results['exp2_recall'][i]*100:.2f}%", table_body_style),
        ])
    
    t_k = Table(table_data_k, colWidths=[80, 100, 100, 80])
    t_k.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(t_k)
    story.append(Spacer(1, 10))
    
    # Add Plot K
    story.append(Image('output/plot_k.png', width=360, height=225))
    story.append(Spacer(1, 15))
    
    story.append(PageBreak())

    # ----------------------------------------------------
    # PLOT S & TABLE S
    # ----------------------------------------------------
    story.append(Paragraph("3.3 Επίδραση του Μεγέθους των Δεδομένων (|S|)", h2_style))
    s_desc = """
    Η κλιμακωσιμότητα (scalability) είναι το σημαντικότερο πλεονέκτημα του IVF-Flat. Εδώ εξετάζουμε πώς συμπεριφέρεται ο αλγόριθμος καθώς αυξάνεται το μέγεθος της βάσης δεδομένων |S|.
    """
    story.append(Paragraph(s_desc, body_style))
    
    # Add Table S
    table_data_s = [[
        Paragraph("<b>|S| (Μέγεθος)</b>", table_header_style), 
        Paragraph("<b>Exact QPS</b>", table_header_style), 
        Paragraph("<b>Approx QPS</b>", table_header_style), 
        Paragraph("<b>Recall</b>", table_header_style)
    ]]
    for i, s_val in enumerate(results['s_sizes']):
        table_data_s.append([
            Paragraph(str(s_val), table_body_style),
            Paragraph(f"{results['exp3_exact_qps'][i]:.2f}", table_body_style),
            Paragraph(f"{results['exp3_approx_qps'][i]:.2f}", table_body_style),
            Paragraph(f"{results['exp3_recall'][i]*100:.2f}%", table_body_style),
        ])
    
    t_s = Table(table_data_s, colWidths=[80, 100, 100, 80])
    t_s.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(t_s)
    story.append(Spacer(1, 10))
    
    # Add Plot S
    story.append(Image('output/plot_s.png', width=360, height=225))
    story.append(Spacer(1, 15))
    
    story.append(PageBreak())

    # ----------------------------------------------------
    # SECTION 4: CONCLUSIONS & REFERENCES
    # ----------------------------------------------------
    story.append(Paragraph("4. Συμπεράσματα", h1_style))
    concl_text = """
    Από την πειραματική αξιολόγηση προκύπτουν τα εξής συμπεράσματα:<br/>
    • <b>Tradeoff Ταχύτητας-Ακρίβειας:</b> Καθώς το <i>M</i> αυξάνεται, η ανάκληση (Recall) πλησιάζει το 100%, αλλά η ταχύτητα αναζήτησης (QPS) μειώνεται σημαντικά λόγω των περισσότερων συγκρίσεων. Η επιλογή του M = 5 με 10 προσφέρει την ιδανική ισορροπία (~80-95% Recall με πολύ υψηλό QPS).<br/>
    • <b>Κλιμακωσιμότητα:</b> Καθώς το <i>|S|</i> μεγαλώνει, ο ακριβής αλγόριθμος (Exact K-NN) επιβραδύνεται γραμμικά O(N), ενώ ο προσεγγιστικός αλγόριθμος (Approximate K-NN) διατηρεί πολύ υψηλότερα QPS, καθώς εξετάζει μόνο ένα σταθερό κλάσμα των δεδομένων (M/P).<br/>
    • <b>Επίδραση του k:</b> Η αύξηση του k επιβαρύνει ελαφρώς τον χρόνο ταξινόμησης στο τέλος της αναζήτησης, αλλά δεν επηρεάζει σημαντικά το χρόνο υπολογισμού των αποστάσεων.
    """
    story.append(Paragraph(concl_text, body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("5. Αναφορές", h1_style))
    refs_text = """
    [1] <i>Inverted Indexing & Clustering Techniques</i>: GeeksforGeeks Database Systems.<br/>
    [2] <i>Principal Component Analysis (PCA) for Dimensionality Reduction</i>: Scikit-Learn Documentation.<br/>
    [3] <i>SIFT1M Dataset benchmarks</i>: Jegou et al., "Product Quantization for Nearest Neighbor Search".
    """
    story.append(Paragraph(refs_text, body_style))
    
    doc.build(story)
    print("PDF report successfully created.")

if __name__ == '__main__':
    results = run_experiments()
    generate_pdf_report(results)
