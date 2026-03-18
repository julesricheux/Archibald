# [archibald 1.0.0](https://github.com/julesricheux/archibald/) :sailboat:✏️

- 🐍 Archibald is a performance prediction package for sailboats and ships in the preliminary design stage. It is written in Python 3.
- 🎓 It is aimed in particular at naval architecture students. It is still under development, but its modules are now ready to be used independently.
- ✏️ In particular, [Rhino](https://www.rhino3d.com) and [AutoCAD](https://www.autodesk.com/products/autocad/overview) models and drawings can be directly imported for computation, giving an easy workflow to quickly assess the impact of design choices on performance, and play with it.

# ⚓ Main features
- Holtrop and Delft hull resistance method
- Built-in mesh-based hydrostatics
- Inviscid fluid-boundary layer coupling

# 🧪 Experimental features
- Sails computation with vertical wind gradient
- Planing hull resistance computation
- Post-stall aerodynamics model

# 📌 Portfolio
Archibald 1.0.0 was widely used in the following student projects:
<table>
    <tr>
        <td width="33%" valign="top">
            <p align="center">
                MOLENEZ II - Sailing general cargo ship for Ponant Islands feeding
            </p>
            <img src="https://github.com/user-attachments/assets/ceba01a2-922a-421d-bd6c-c827573c1b30" alt="Molenez II Render">
            <img src="https://github.com/user-attachments/assets/db25fa33-8740-48de-9531-691ed062cc27" alt="Molenez II Sail plan">
            <img src="https://github.com/user-attachments/assets/e0e79c12-b445-46c4-895b-0544a536c75d" alt="Molenez II Stability">
        </td>
        <td width="33%" valign="top">
            <p align="center">
                NEREUS - Fast humanitarian aid vessel (tested in Centrale Nantes' towing tank)
            </p>
            <img src="https://github.com/user-attachments/assets/d63d57ea-cb07-4af6-bede-40f578e1951e" alt="Nereus render">
            <img src="https://github.com/user-attachments/assets/9f410fe0-0764-407c-9208-8b29de3075d3" alt="Nereus towing tank">
            <img src="https://github.com/user-attachments/assets/ea89a466-430e-4abe-b30e-adcf57495677" alt="Nereus Stability">
        </td>
        <td width="33%" valign="top">
            <p align="center">
                PAKI'AKA - Radical and high-performance bamboo 18ft skiff
            </p>
            <img src="https://github.com/user-attachments/assets/1d9cca6a-bce4-4243-a564-03f7da629c61" alt="Paki'aka render 1">
            <img src="https://github.com/user-attachments/assets/cd04bdaa-1c12-40e7-ba01-869ccf1edff9" alt="Paki'aka render 1">
        </td>
    </tr>
</table>

# 💡 Examples

Different examples of basic use can be found in /examples/:
- Hydrostatics and stability curve computation
- Hull resistance computation
- Appendage hydrodynamics computation
- Sails aerodynamics computation

# 📜 References
Papers used to write this code can be found in /refs/

# 🔗 Dependencies
## ⚓ For general use:
NumPy
SciPy
Trimesh
ezdxf
Shapely
csv
tqdm
Matplotlib

## 🧪 For experimental features:
AeroSandbox
OpenPlaning

# ⚖️ License
Licensed under [MIT License, terms here](https://opensource.org/license/mit)
