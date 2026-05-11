import re

f = open('F:/datasetpic/templates/index.html', 'r', encoding='utf-8')
content = f.read()
f.close()

key = "currentLabels['\u89d2\u8272\u540d\u79f0']"

print("=== Diagnosis ===")
print(f"selectRole functions: {content.count('function selectRole')}")
print(f"removeRole functions: {content.count('function removeRole')}")
print(f"searchRoles functions: {content.count('function searchRoles')}")
print(f"renderSelectedRoles functions: {content.count('function renderSelectedRoles')}")
print(f"currentLabels role key refs: {content.count(key)}")
print(f"saveCurrent has 'labels: currentLabels': {'labels: currentLabels' in content}")
print(f"loadAnnotation has 'currentLabels = ann.labels': {'currentLabels = ann.labels' in content}")

# Check selectImage overwriting
line433 = "currentLabels = img.labels ? JSON.parse(JSON.stringify(img.labels)) : {}"
print(f"selectImage overwrites currentLabels: {line433 in content}")

# Check if toggleTag calls renderLabelPanel (which resets the role search box)
print(f"toggleTag calls renderLabelPanel: {'renderLabelPanel()' in content}")

print("\n=== Key finding ===")
print("selectRole updates currentLabels[key] = selectedRoles")
print("saveCurrent sends { labels: currentLabels } to server")
print("Server saves labels including 角色名称")
print("After save, selectImage -> loadAnnotation restores from server")
print("renderRoleSearch reads currentLabels[key] to restore selectedRoles")
print("\nThe flow looks correct. Issue might be:")
print("1. User not clicking Save after selecting roles")
print("2. Or toggleTag -> renderLabelPanel resets selectedRoles before save")

# Count how many times renderLabelPanel is called
calls = len(re.findall(r'renderLabelPanel\(\)', content))
print(f"\nrenderLabelPanel() is called {calls} times in total")
print("Each call rebuilds the entire label panel including role search")
print("But renderRoleSearch reads currentLabels[key] so data should be preserved")
