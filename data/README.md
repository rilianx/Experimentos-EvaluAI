# Datos cifrados

Los datasets contienen respuestas de estudiantes y el repositorio es público, por lo que solo se versionan cifrados.

Para cifrar (por ejemplo, desde Git Bash):

```bash
gpg --symmetric --cipher-algo AES256 --output data/datasets_v2.xlsx.gpg datasets_v2.xlsx
```

La contraseña se guarda como secret `DATASET_PASSPHRASE` del repositorio (Settings → Secrets and variables → Actions).
El workflow `Experimentos` descifra los archivos `data/*.gpg` en el runner.

Para descifrar localmente:

```bash
gpg --output data/datasets_v2.xlsx --decrypt data/datasets_v2.xlsx.gpg
```

Las salidas completas de cada corrida se publican como artifact `runs-encrypted-<id>` cifrado con la misma contraseña:

```bash
gpg --output runs.tar.gz --decrypt runs.tar.gz.gpg && tar xzf runs.tar.gz
```
