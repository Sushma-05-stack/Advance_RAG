import os
import json

os.makedirs('sample_docs/medical', exist_ok=True)

diseases = [
    ('Asthma', 'A respiratory condition marked by spasms in the bronchi of the lungs, causing difficulty in breathing. Treatment includes inhaled corticosteroids.'),
    ('Diabetes', 'A metabolic disease in which the body\'s inability to produce any or enough insulin causes elevated levels of glucose in the blood. Management includes insulin therapy and diet control.'),
    ('Hypertension', 'Abnormally high blood pressure. Treatment often includes ACE inhibitors, calcium channel blockers, and lifestyle changes.'),
    ('Tuberculosis', 'An infectious bacterial disease characterized by the growth of nodules in the tissues, especially the lungs. Treatment requires a long course of multiple antibiotics.'),
    ('Influenza', 'A highly contagious viral infection of the respiratory passages causing fever, severe aching, and catarrh. Prevented by annual vaccination.'),
    ('COVID-19', 'A contagious respiratory and vascular disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).'),
    ('Cancer', 'A disease caused by an uncontrolled division of abnormal cells in a part of the body. Treatments include chemotherapy, radiation, and surgery.'),
    ('Alzheimer\'s', 'Progressive mental deterioration that can occur in middle or old age, due to generalized degeneration of the brain.'),
    ('Parkinson\'s', 'A progressive disease of the nervous system marked by tremor, muscular rigidity, and slow, imprecise movement.'),
    ('Malaria', 'An intermittent and remittent fever caused by a protozoan parasite that invades the red blood cells, transmitted by mosquitoes in many tropical and subtropical regions.'),
    ('Cholera', 'An infectious and often fatal bacterial disease of the small intestine, typically contracted from infected water supplies.'),
    ('Dengue fever', 'A mosquito-borne tropical disease caused by the dengue virus. Symptoms include high fever, headache, vomiting, muscle and joint pains.'),
    ('Zika virus', 'A virus transmitted by mosquitoes which typically causes asymptomatic or mild infection (fever and rash) in humans.'),
    ('Ebola', 'An infectious and generally fatal disease marked by fever and severe internal bleeding, spread through contact with infected body fluids.'),
    ('HIV/AIDS', 'A chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV), which interferes with the body\'s ability to fight infections.'),
    ('Hepatitis B', 'A severe form of viral hepatitis transmitted in infected blood, causing fever, debility, and jaundice.'),
    ('Hepatitis C', 'A form of viral hepatitis transmitted in infected blood, causing chronic liver disease.'),
    ('Pneumonia', 'Lung inflammation caused by bacterial or viral infection, in which the air sacs fill with pus and may become solid.'),
    ('COPD', 'Chronic obstructive pulmonary disease, a type of obstructive lung disease characterized by long-term breathing problems and poor airflow.'),
    ('Stroke', 'A sudden disabling attack or loss of consciousness caused by an interruption in the flow of blood to the brain.'),
    ('Coronary artery disease', 'Impedance or blockage of one or more arteries that supply blood to the heart, usually due to atherosclerosis.'),
    ('Heart failure', 'Severe failure of the heart to function properly, especially as a cause of death.'),
    ('Arrhythmia', 'A condition in which the heart beats with an irregular or abnormal rhythm.'),
    ('Anemia', 'A condition marked by a deficiency of red blood cells or of hemoglobin in the blood, resulting in pallor and weariness.'),
    ('Leukemia', 'A malignant progressive disease in which the bone marrow and other blood-forming organs produce increased numbers of immature or abnormal leukocytes.'),
    ('Lymphoma', 'Cancer of the lymph nodes.'),
    ('Multiple sclerosis', 'A chronic, typically progressive disease involving damage to the sheaths of nerve cells in the brain and spinal cord.'),
    ('Rheumatoid arthritis', 'A chronic progressive disease causing inflammation in the joints and resulting in painful deformity and immobility.'),
    ('Osteoarthritis', 'Degeneration of joint cartilage and the underlying bone, most common from middle age onward.'),
    ('Gout', 'A disease in which defective metabolism of uric acid causes arthritis, especially in the smaller bones of the feet.'),
    ('Migraine', 'A recurrent throbbing headache that typically affects one side of the head and is often accompanied by nausea and disturbed vision.'),
    ('Epilepsy', 'A neurological disorder marked by sudden recurrent episodes of sensory disturbance, loss of consciousness, or convulsions.'),
    ('Schizophrenia', 'A long-term mental disorder of a type involving a breakdown in the relation between thought, emotion, and behavior.'),
    ('Bipolar disorder', 'A mental disorder marked by alternating periods of elation and depression.'),
    ('Major depressive disorder', 'A mental disorder characterized by at least two weeks of pervasive low mood, low self-esteem, and loss of interest.'),
    ('Anxiety disorder', 'A mental health disorder characterized by feelings of worry, anxiety, or fear that are strong enough to interfere with one\'s daily activities.'),
    ('Autism', 'A developmental disorder of variable severity that is characterized by difficulty in social interaction and communication and by restricted or repetitive patterns of thought and behavior.'),
    ('ADHD', 'Attention-deficit/hyperactivity disorder, a chronic condition including attention difficulty, hyperactivity, and impulsiveness.'),
    ('Obesity', 'The condition of being grossly fat or overweight.'),
    ('Malnutrition', 'Lack of proper nutrition, caused by not having enough to eat, not eating enough of the right things, or being unable to use the food that one does eat.'),
    ('Vitamin D deficiency', 'A state in which the body has inadequate vitamin D.'),
    ('Scurvy', 'A disease caused by a deficiency of vitamin C, characterized by swollen bleeding gums and the opening of previously healed wounds.'),
    ('Rickets', 'A disease of children caused by vitamin D deficiency, characterized by imperfect calcification, softening, and distortion of the bones typically resulting in bow legs.'),
    ('Osteoporosis', 'A medical condition in which the bones become brittle and fragile from loss of tissue, typically as a result of hormonal changes, or deficiency of calcium or vitamin D.'),
    ('Hyperthyroidism', 'Overactivity of the thyroid gland, resulting in a rapid heartbeat and an increased rate of metabolism.'),
    ('Hypothyroidism', 'Abnormally low activity of the thyroid gland, resulting in retardation of growth and mental development in children and adults.'),
    ('Celiac disease', 'A disease in which the small intestine is hypersensitive to gluten, leading to difficulty in digesting food.'),
    ('Crohn\'s disease', 'A chronic inflammatory disease of the intestines, especially the colon and ileum, associated with ulcers and fistulae.'),
    ('Ulcerative colitis', 'An inflammatory bowel disease (IBD) that causes long-lasting inflammation and ulcers in the digestive tract.'),
    ('Appendicitis', 'A serious medical condition in which the appendix becomes inflamed and painful.')
]

eval_dataset = []

for idx, (disease, desc) in enumerate(diseases):
    filename = disease.replace(' ', '_').replace('/', '_') + '.txt'
    content = f"# {disease}\n\n## Overview\n{desc}\n\n## Diagnosis\nDiagnosis of {disease} typically involves clinical assessment, patient history, and specific laboratory tests or imaging. Early detection is crucial for effective management.\n\n## Treatment\nTreatment options vary depending on the severity but generally aim to manage symptoms and improve the patient's quality of life. Patients should consult healthcare providers for tailored treatment plans."
    
    with open(f'sample_docs/medical/{filename}', 'w', encoding='utf-8') as f:
        f.write(content)
        
    eval_dataset.append({
        'question': f'What is the standard description and treatment approach for {disease}?',
        'ground_truth': desc
    })

# Write the evaluation dataset
with open('eval_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(eval_dataset, f, indent=2)

print(f'Generated {len(diseases)} medical documents and eval_dataset.json')
