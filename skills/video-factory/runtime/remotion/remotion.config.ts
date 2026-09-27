import {Config} from '@remotion/cli/config';

// The Python prepare-render command materializes project assets under work/.
// The CLI's --public-dir flag is a URL prefix, so the filesystem directory is
// supplied through this config hook instead.
Config.setPublicDir(process.env.VIDEO_FACTORY_PUBLIC_DIR ?? './public');
