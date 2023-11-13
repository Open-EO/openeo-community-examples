# Onnx example

https://onnx.ai/ allows users to save e.g. their tensorflow or pytorch model to a standard file format. They can then execute that model using the onnxruntime. This way openEO UDFs can support inference of tensorflow and pytorch models without having to upload their dependencies. 

The https://artifactory.vgt.vito.be:443/auxdata-public/openeo/onnx_dependencies.zip file was created by installing the onnxruntime and its dependencies into a virtual environment and then zipping only the required library directories.
```
python3.8 -m venv venv
source venv/bin/activate
pip install onnxruntime
cd venv/lib/python3.8/site-packages/
zip -r onnx_dependencies.zip *  # But exclude the standard libraries that were present before installing onnx.
```