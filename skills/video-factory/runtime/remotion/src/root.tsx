import React from 'react';
import {
  AbsoluteFill,
  CanvasImage,
  Composition,
  Sequence,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {Audio, Video} from '@remotion/media';

type Segment = {
  id?: string;
  type: string;
  source_url?: string;
  source_in?: number;
  source_out?: number;
  timeline_start: number;
  timeline_end: number;
  text?: string;
  title?: string;
  subtitle?: {text?: string};
  annotations?: Array<{text: string; position?: string}>;
  crop?: {
    object_position?: string;
    fit?: 'cover' | 'contain';
    zoom_start?: number;
    zoom_end?: number;
    '16:9'?: {object_position?: string; fit?: 'cover' | 'contain'; zoom_start?: number; zoom_end?: number};
    '9:16'?: {object_position?: string; fit?: 'cover' | 'contain'; zoom_start?: number; zoom_end?: number};
  };
  audio?: {volume?: number; preserve_natural?: boolean};
  volume?: number;
  fade_in_seconds?: number;
  fade_out_seconds?: number;
  style?: {background?: string; color?: string};
};

type RenderProps = {
  title: string;
  width: number;
  height: number;
  fps: number;
  duration_seconds: number;
  timeline: Segment[];
};

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

const secondsToFrames = (seconds: number, fps: number) =>
  Math.max(0, Math.round(seconds * fps));

const segmentFrames = (segment: Segment, fps: number) =>
  Math.max(1, secondsToFrames(segment.timeline_end, fps) - secondsToFrames(segment.timeline_start, fps));

const mediaUrl = (url: string) => staticFile(url.replace(/^\/+/, ''));

const audioVolume = (segment: Segment, fps: number, localFrame: number, duckingRanges: ReadonlyArray<readonly [number, number]> = []) => {
  const durationFrames = segmentFrames(segment, fps);
  const defaultFade = segment.type === 'music' ? 0.25 : 0.1;
  const fadeInFrames = secondsToFrames(segment.fade_in_seconds ?? defaultFade, fps);
  const fadeOutFrames = secondsToFrames(segment.fade_out_seconds ?? defaultFade, fps);
  const fadeInGain = fadeInFrames > 0 ? clamp(localFrame / fadeInFrames, 0, 1) : 1;
  const fadeOutGain = fadeOutFrames > 0 ? clamp((durationFrames - localFrame) / fadeOutFrames, 0, 1) : 1;
  const time = segment.timeline_start + localFrame / fps;
  const duckGain = segment.type === 'music'
    ? duckingRanges.reduce((lowest, [start, end]) => {
        const fade = 0.25;
        const quiet = 0.24;
        if (time >= start && time < end) return Math.min(lowest, quiet);
        if (time >= start - fade && time < start) {
          return Math.min(lowest, 1 - ((time - (start - fade)) / fade) * (1 - quiet));
        }
        if (time >= end && time < end + fade) {
          return Math.min(lowest, quiet + ((time - end) / fade) * (1 - quiet));
        }
        return lowest;
      }, 1)
    : 1;
  if (segment.type === 'video' && segment.audio?.preserve_natural === false) return 0;
  const baseVolume = segment.audio?.volume ?? segment.volume ?? (segment.type === 'music' ? 0.2 : 1);
  return baseVolume * Math.min(fadeInGain, fadeOutGain) * duckGain;
};

const segmentText = (segment: Segment) =>
  segment.text ?? segment.title ?? segment.subtitle?.text ?? segment.annotations?.[0]?.text ?? '';

const FadeLayer: React.FC<{segment: Segment; children: React.ReactNode}> = ({
  segment,
  children,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const duration = segmentFrames(segment, fps);
  const fadeIn = secondsToFrames(segment.fade_in_seconds ?? 0.2, fps);
  const fadeOut = secondsToFrames(segment.fade_out_seconds ?? 0.2, fps);
  const fromStart = fadeIn > 0 ? clamp(frame / fadeIn, 0, 1) : 1;
  const fromEnd = fadeOut > 0 ? clamp((duration - frame) / fadeOut, 0, 1) : 1;
  return <AbsoluteFill style={{opacity: Math.min(fromStart, fromEnd)}}>{children}</AbsoluteFill>;
};

const MediaLayer: React.FC<{segment: Segment}> = ({segment}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const length = segmentFrames(segment, fps);
  const ratioCrop = height > width ? segment.crop?.['9:16'] : segment.crop?.['16:9'];
  const crop = {...segment.crop, ...ratioCrop};
  const zoomStart = crop?.zoom_start ?? 1;
  const zoomEnd = crop?.zoom_end ?? zoomStart;
  const zoom = zoomStart + (zoomEnd - zoomStart) * clamp(frame / Math.max(1, length - 1), 0, 1);
  const style: React.CSSProperties = {
    width: '100%',
    height: '100%',
    objectPosition: crop?.object_position ?? '50% 50%',
    scale: zoom,
  };
  if (!segment.source_url) return null;
  return (
    <FadeLayer segment={segment}>
      {segment.type === 'video' ? (
        <Video
          src={mediaUrl(segment.source_url)}
          trimBefore={secondsToFrames(segment.source_in ?? 0, fps)}
          trimAfter={segment.source_out === undefined ? undefined : secondsToFrames(segment.source_out, fps)}
          volume={(localFrame) => audioVolume(segment, fps, localFrame)}
          objectFit={crop?.fit ?? 'cover'}
          style={style}
        />
      ) : (
        <CanvasImage src={mediaUrl(segment.source_url)} style={{...style, objectFit: crop?.fit ?? 'cover'}} />
      )}
    </FadeLayer>
  );
};

const TextLayer: React.FC<{segment: Segment}> = ({segment}) => {
  const {width, height} = useVideoConfig();
  const text = segmentText(segment);
  const isVertical = height > width;
  const isSubtitle = segment.type === 'subtitle';
  const isTitle = segment.type === 'title' || segment.type === 'text_card';
  const position = segment.annotations?.[0]?.position ?? 'center';
  const common: React.CSSProperties = {
    color: segment.style?.color ?? '#fff',
    fontFamily: 'Arial, sans-serif',
    textShadow: isSubtitle ? '0 2px 8px rgba(0,0,0,.9)' : '0 2px 10px rgba(0,0,0,.38)',
    textAlign: 'center',
    whiteSpace: 'pre-wrap',
    overflowWrap: 'break-word',
  };
  if (isSubtitle) {
    return (
      <AbsoluteFill
        style={{
          ...common,
          justifyContent: 'flex-end',
          alignItems: 'center',
          padding: '0 ' + (isVertical ? '8%' : '10%') + ' ' + (isVertical ? '14%' : '8%'),
          fontSize: Math.round(width * (isVertical ? 0.052 : 0.036)),
          fontWeight: 700,
          lineHeight: 1.35,
        }}
      >
        <div style={{maxWidth: '100%', background: 'rgba(0,0,0,.38)', padding: '0.15em 0.35em'}}>
          {text}
        </div>
      </AbsoluteFill>
    );
  }
  if (isTitle || segment.type === 'annotation' || segment.type === 'overlay') {
    const centered = isTitle || position === 'center';
    return (
      <AbsoluteFill
        style={{
          ...common,
          background: isTitle ? (segment.style?.background ?? '#183044') : 'transparent',
          justifyContent: centered ? 'center' : 'flex-end',
          alignItems: 'center',
          padding: '8%',
          fontSize: Math.round(width * (isTitle ? 0.064 : 0.042)),
          fontWeight: isTitle ? 700 : 600,
        }}
      >
        {text}
      </AbsoluteFill>
    );
  }
  return null;
};

const Timeline: React.FC<RenderProps> = (props) => {
  const {fps} = useVideoConfig();
  const visualTypes = new Set(['photo', 'video', 'title', 'text_card', 'subtitle', 'overlay', 'annotation']);
  const audioTypes = new Set(['music', 'narration', 'natural_audio']);
  const duckingRanges = props.timeline
    .filter(
      (segment) =>
        segment.type === 'narration' ||
        segment.type === 'natural_audio' ||
        (segment.type === 'video' && segment.audio?.preserve_natural === true),
    )
    .map((segment) => [segment.timeline_start, segment.timeline_end] as const);
  return (
    <AbsoluteFill style={{backgroundColor: '#101820'}}>
      {props.timeline.map((segment, index) => {
        const from = secondsToFrames(segment.timeline_start, fps);
        const duration = segmentFrames(segment, fps);
        if (visualTypes.has(segment.type)) {
          return (
            <Sequence key={segment.id ?? segment.type + '-' + index} from={from} durationInFrames={duration} name={segment.id}>
              {segment.type === 'photo' || segment.type === 'video' ? (
                <MediaLayer segment={segment} />
              ) : (
                <FadeLayer segment={{...segment, fade_in_seconds: segment.fade_in_seconds ?? 0, fade_out_seconds: segment.fade_out_seconds ?? 0}}>
                  <TextLayer segment={segment} />
                </FadeLayer>
              )}
            </Sequence>
          );
        }
        if (audioTypes.has(segment.type) && segment.source_url) {
          return (
            <Sequence key={segment.id ?? segment.type + '-' + index} from={from} durationInFrames={duration} name={segment.id}>
              <Audio
                src={mediaUrl(segment.source_url)}
                volume={(localFrame) => audioVolume(segment, fps, localFrame, duckingRanges)}
                trimBefore={secondsToFrames(segment.source_in ?? 0, fps)}
                trimAfter={segment.source_out === undefined ? undefined : secondsToFrames(segment.source_out, fps)}
              />
            </Sequence>
          );
        }
        return null;
      })}
    </AbsoluteFill>
  );
};

export const VideoFactoryRoot: React.FC = () => (
  <Composition
    id="VideoFactory"
    component={Timeline}
    width={1920}
    height={1080}
    fps={30}
    durationInFrames={300}
    calculateMetadata={({props}: {props: RenderProps}) => ({
      width: props.width,
      height: props.height,
      fps: props.fps,
      durationInFrames: Math.max(1, secondsToFrames(props.duration_seconds, props.fps)),
    })}
    defaultProps={{
      title: 'Video Factory',
      width: 1920,
      height: 1080,
      fps: 30,
      duration_seconds: 10,
      timeline: [],
    }}
  />
);
