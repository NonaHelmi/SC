import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# تعریف متغیرهای ورودی و خروجی
attendance = ctrl.Antecedent(np.arange(0, 101, 1), 'attendance')
homework = ctrl.Antecedent(np.arange(0, 101, 1), 'homework')
participation = ctrl.Antecedent(np.arange(0, 101, 1), 'participation')
grade = ctrl.Consequent(np.arange(0, 101, 1), 'grade')

# تعریف توابع عضویت برای attendance
attendance['low'] = fuzz.trimf(attendance.universe, [0, 0, 50])
attendance['high'] = fuzz.trimf(attendance.universe, [50, 100, 100])

# تعریف توابع عضویت برای homework
homework['low'] = fuzz.trimf(homework.universe, [0, 0, 50])
homework['high'] = fuzz.trimf(homework.universe, [50, 100, 100])

# تعریف توابع عضویت برای participation
participation['low'] = fuzz.trimf(participation.universe, [0, 0, 50])
participation['high'] = fuzz.trimf(participation.universe, [50, 100, 100])

# تعریف توابع عضویت برای grade
grade['low'] = fuzz.trimf(grade.universe, [0, 0, 50])
grade['high'] = fuzz.trimf(grade.universe, [50, 100, 100])

# تعریف قواعد فازی
rule1 = ctrl.Rule(attendance['low'] & homework['low'] & participation['low'], grade['low'])
rule2 = ctrl.Rule(attendance['high'] | homework['high'] | participation['high'], grade['high'])

# اضافه کردن قواعد به سیستم فازی
grade_ctrl = ctrl.ControlSystem([rule1, rule2])
grade_evaluator = ctrl.ControlSystemSimulation(grade_ctrl)

# ارزیابی با ورودی‌ها
grade_evaluator.input['attendance'] = 75
grade_evaluator.input['homework'] = 60
grade_evaluator.input['participation'] = 80
grade_evaluator.compute()

# گرفتن خروجی
print(grade_evaluator.output['grade'])
