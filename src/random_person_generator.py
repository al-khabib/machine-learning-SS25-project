import random

# Define ranges and categories based on the dataset features
age_range = (18, 65)  # typical adult age range
gender_categories = ['M', 'F']
height_range = (150, 190)  # in cm
weight_range = (45, 100)  # in kg
body_fat_range = (10, 40)  # in %
diastolic_range = (60, 100)  # typical diastolic BP
systolic_range = (90, 160)  # typical systolic BP
gripForce_range = (15, 60)  # approximate grip force
sit_bend_range = (0, 30)  # sit and bend forward in cm
situps_range = (0, 70)  # sit-ups counts
broad_jump_range = (90, 280)  # broad jump in cm

# Generate random values within these ranges
random_person = {
    'age': random.randint(*age_range),
    'gender': random.choice(gender_categories),
    'height_cm': round(random.uniform(*height_range), 1),
    'weight_kg': round(random.uniform(*weight_range), 1),
    'body fat_%': round(random.uniform(*body_fat_range), 1),
    'diastolic': random.randint(*diastolic_range),
    'systolic': random.randint(*systolic_range),
    'gripForce': round(random.uniform(*gripForce_range), 1),
    'sit and bend forward_cm': round(random.uniform(*sit_bend_range), 1),
    'sit-ups counts': random.randint(*situps_range),
    'broad jump_cm': random.randint(*broad_jump_range)
}
random_person
