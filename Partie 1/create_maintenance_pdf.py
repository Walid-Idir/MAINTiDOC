"""
Générateur d'un document technique dans le Domaine de la maintenance des machines :
'Manuel de Maintenance Préventive et Corrective - Compresseur Industriel à Vis (Modèle CPX-450)'
Contient du texte technique, des tableaux de maintenance et des schémas/images.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = BASE_DIR / "manuel_maintenance_compresseur_industriel.pdf"


def generate_maintenance_doc():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1B365D'),
        alignment=1,
        spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4A5568'),
        alignment=1,
        spaceAfter=25
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0D47A1'),
        spaceBefore=15,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=8
    )
    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Italic'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#4A5568'),
        alignment=1,
        spaceBefore=5,
        spaceAfter=15
    )

    story = []

    # ---------------------------------------------------------
    # Page 1 : Présentation Générale et Sécurité
    # ---------------------------------------------------------
    story.append(Paragraph("MANUEL TECHNIQUE DE MAINTENANCE DES MACHINES", title_style))
    story.append(Paragraph("Compresseur Industriel à Vis Lubrifiée — Série CPX-450 (Pression 10 Bar - 55 kW)", subtitle_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Consignes Générales de Sécurité et Fonctionnement", h2_style))
    intro_p = (
        "Le compresseur à vis série CPX-450 est un équipement électromécanique lourd dédié à la fourniture continue "
        "d'air comprimé haute pression pour les ateliers d'usinage et de fabrication. Avant toute intervention technique "
        "ou opération de maintenance sur l'étage de compression ou le moteur triphasé, il est impératif de consigner "
        "les énergies (cadenassage électrique LOTO, dépressurisation complète du circuit pneumatique et vérification de "
        "la température du carter d'huile). Le non-respect des procédures de consignation peut entraîner des accidents "
        "mécaniques graves ou des brûlures thermiques."
    )
    story.append(Paragraph(intro_p, body_style))

    story.append(Paragraph("2. Schéma Mécanique des Composants Principaux", h2_style))
    
    # Dessin vectoriel du schéma de la machine
    drawing = Drawing(520, 140)
    drawing.add(Rect(10, 10, 500, 120, fillColor=colors.HexColor('#F7FAFC'), strokeColor=colors.HexColor('#CBD5E0'), strokeWidth=1))
    # Moteur
    drawing.add(Rect(30, 30, 100, 80, fillColor=colors.HexColor('#E2E8F0'), strokeColor=colors.HexColor('#4A5568'), strokeWidth=2))
    drawing.add(String(55, 65, "MOTEUR 55 kW", fontSize=9, fontName='Helvetica-Bold', fillColor=colors.HexColor('#1A202C')))
    # Arbre / Accouplement
    drawing.add(Line(130, 70, 180, 70, strokeColor=colors.HexColor('#2B6CB0'), strokeWidth=6))
    drawing.add(String(135, 80, "Accouplement", fontSize=7, fontName='Helvetica', fillColor=colors.HexColor('#2B6CB0')))
    # Étage de vis
    drawing.add(Rect(180, 25, 120, 90, fillColor=colors.HexColor('#BEE3F8'), strokeColor=colors.HexColor('#2B6CB0'), strokeWidth=2))
    drawing.add(String(195, 75, "BLOC COMPRESSEUR", fontSize=8, fontName='Helvetica-Bold', fillColor=colors.HexColor('#1A365D')))
    drawing.add(String(205, 55, "Rotor Mâle/Femelle", fontSize=7, fontName='Helvetica', fillColor=colors.HexColor('#2C5282')))
    # Séparateur d'huile
    drawing.add(Circle(370, 70, 45, fillColor=colors.HexColor('#FED7D7'), strokeColor=colors.HexColor('#C53030'), strokeWidth=2))
    drawing.add(String(335, 75, "RÉSERVOIR DÉshuileur", fontSize=7, fontName='Helvetica-Bold', fillColor=colors.HexColor('#742A2A')))
    drawing.add(String(345, 60, "Filtre Coalescent", fontSize=7, fontName='Helvetica', fillColor=colors.HexColor('#9B2C2C')))
    # Tuyauterie
    drawing.add(Line(300, 70, 325, 70, strokeColor=colors.HexColor('#C53030'), strokeWidth=4))
    drawing.add(Line(415, 70, 480, 70, strokeColor=colors.HexColor('#38A169'), strokeWidth=5))
    drawing.add(String(425, 80, "Air Sortie 10 bar", fontSize=7, fontName='Helvetica-Bold', fillColor=colors.HexColor('#22543D')))
    
    story.append(drawing)
    story.append(Paragraph("Figure 1: Schéma cinématique et flux des fluides du compresseur industriel CPX-450", caption_style))

    # ---------------------------------------------------------
    # Tableau 1 : Plan de Maintenance Préventive
    # ---------------------------------------------------------
    story.append(Paragraph("3. Calendrier des Interventions de Maintenance Préventive", h2_style))
    table1_data = [
        ["Composant Machine", "Action de Maintenance", "Périodicité", "Spécification / Pièce"],
        ["Filtre à air d'aspiration", "Nettoyage / Remplacement", "500 h / 2 000 h", "Cartouche plissée Réf. FA-450"],
        ["Huile de synthèse compresseur", "Vidange et analyse viscosité", "4 000 heures", "Huile ISO VG 46 Synthétique (25 L)"],
        ["Filtre séparateur d'huile", "Remplacement cartouche", "4 000 heures", "Séparateur coalescent Réf. SP-88"],
        ["Courroies de transmission", "Contrôle tension et flèche", "1 000 heures", "Tension au tensiomètre: 450 N"],
        ["Roulements de palier moteur", "Graissage haute température", "2 500 heures", "Graisse Polyurée SKF LGHP 2 (15 g)"],
        ["Soupape de sécurité tarée", "Test d'ouverture manuelle", "Mensuel", "Pression d'échappement : 11.5 bar"]
    ]
    t1 = Table(table1_data, colWidths=[130, 130, 90, 170])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t1)
    story.append(Paragraph("Table 1: Programme d'interventions préventives systématiques et consommables recommandés", caption_style))

    # ---------------------------------------------------------
    # Tableau 2 : Couples de Serrage et Tolérances Mécaniques
    # ---------------------------------------------------------
    story.append(Paragraph("4. Couples de Serrage et Métrologie Mécanique", h2_style))
    table2_data = [
        ["Sous-ensemble Mécanique", "Type de Boulonnerie", "Couple Recommandé (N·m)", "Graissage Filetage"],
        ["Flasque d'aspiration vis", "M12 Classe 8.8", "85 N·m", "Huile moteur légère"],
        ["Culasse étage haute pression", "M16 Classe 10.9", "210 N·m", "Pâte anti-grippante MoS2"],
        ["Fixation socle châssis / plot antivibratile", "M20 Classe 8.8", "380 N·m", "Montage à sec"],
        ["Capteur de pression piézoélectrique", "G 1/4 Gaz", "25 N·m", "Ruban PTFE téflon"]
    ]
    t2 = Table(table2_data, colWidths=[150, 110, 120, 140])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0D47A1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    story.append(t2)
    story.append(Paragraph("Table 2: Spécifications de serrage dynamométrique et tolérances de montage", caption_style))

    doc.build(story)
    print(f"[✓] Document technique de maintenance généré : {OUTPUT_PDF}")


if __name__ == "__main__":
    generate_maintenance_doc()
