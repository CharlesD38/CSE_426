
from traitlets.config import Config
import nbformat as nbf
from nbconvert.exporters import PDFExporter
from nbconvert.preprocessors import TagRemovePreprocessor

# Setup config
c = Config()

c.TagRemovePreprocessor.remove_input_tags = ('hide_input',)
c.TagRemovePreprocessor.enabled = True
# Configure and run out exporter
c.PDFExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]

exporter = PDFExporter(config=c)
exporter.register_preprocessor(TagRemovePreprocessor(config=c),True)

# Configure and run our exporter - returns a tuple - first element with html,
# second with notebook metadata
output = PDFExporter(config=c).from_filename(
        r"C:\MFE\MFE Sem 3\CSE 426\CSE_426\HW10\867002105_HW10.ipynb")

# Write to output html file
with open(r"C:\MFE\MFE Sem 3\CSE 426\CSE_426\HW10\867002105_HW10.pdf",  "wb") as f:
    f.write(output[0])
    