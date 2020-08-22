# Components structure

Each category of components offerd by NLU is divided into one folder.
In each category folder, each individual component has its own folder.
In each components individual folder there is a component_infos.json file with crucial information about how to handle the component internally in NLU.
Additionaly, each components individual folder provided a Python script for creating the component.


In the Python Scripts in the directory nlu/components act as accessor to each component type.
These accessors take in string identifiers, which have to map to the nams in the component_infos.json file, or it will result in a Component not fould exception.

## Components structure :
To ad a new component either create a new catagory and then a foler with the required files in it or add your component folder to an already existing category of components.

